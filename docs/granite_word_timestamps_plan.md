# Plan: Add Word-Level Timestamps to Granite Transcriber

## Summary
Implement word-level timestamp support for IBM Granite Speech models using **attention-based temporal alignment**. This method extracts word-level timestamps from Granite's internal attention patterns by training a small auxiliary model on attention matrix activations.

## Approach: Attention-Based Alignment (Granite-Specific)

Based on research from the community (see `transcripts/granite_speech_transcript.txt`), we can extract word-level timestamps from Granite by:

1. **Extracting attention matrices** during Granite inference
2. **Identifying monotonic attention heads** (later text tokens attend to later audio tokens)
3. **Training a small auxiliary model** to predict word start times from attention patterns

This method is **superior to generic forced alignment** because:
- ✅ Leverages Granite's internal knowledge (already learned during training)
- ✅ More accurate than external CTC alignment models
- ✅ Custom-built for Granite's architecture
- ✅ Proven to "work incredibly well" in practice

## Technical Background

### How the Method Works

**Step 1: Granite's Architecture Context**
- Conformer encoder (CTC-based) → outputs at 50 Hz
- Q-former downsamples to 10 Hz (10 embeddings per second of audio)
- Each audio token represents 0.1 seconds (100ms stride)
- Granite LLM with LoRA adapter generates text from audio embeddings

**Step 2: Attention Pattern Discovery**
- Granite's cross-attention between audio tokens and text tokens reveals temporal alignment
- Certain attention heads show **monotonic patterns**: later text tokens attend to later audio tokens
- Example heads found: Layer 16 Head 23, Layer 34 Heads 5 & 29, Layer 33 Head 21, etc.

**Step 3: Auxiliary Model Training**
- Select top 10 most monotonic attention heads
- For each word start token, extract a 10D feature vector (one value per head per audio token)
- Train small transformer (8 heads, 4 layers, 256 dim, RoPE) to predict word start times
- Loss: MSE between predicted and ground truth (from LibriSpeech corpus)
- Output: Softmax over audio tokens → weighted average gives predicted start time

### Why This Works

The method exploits the fact that models like Granite **internally learn temporal alignment** even though they're not explicitly trained for it (similar to OpenAI's unsupervised sentiment neuron discovery). By analyzing which attention heads have learned this alignment, we can extract it.

## Implementation Strategy

### Phase-Based Approach

This is a complex multi-phase implementation. We'll build it incrementally:

**Phase 1**: Extract attention matrices from Granite (**Foundation**)
**Phase 2**: Analyze attention patterns & identify monotonic heads (**Discovery**)
**Phase 3**: Collect training data from LibriSpeech (**Data Collection**)
**Phase 4**: Train auxiliary alignment model (**Training**)
**Phase 5**: Integrate into production pipeline (**Deployment**)

Each phase builds on the previous one, allowing for testing and validation along the way.

## Detailed Implementation Plan

---

### **PHASE 1: Extract Attention Matrices from Granite**

**Goal**: Modify Granite inference to capture attention matrices

**Files to Create/Modify**:
- `src/granite_attention_extractor.py` (NEW)
- `src/transcribe_audio.py` (modify GraniteTranscriber)

**Implementation Steps**:

1. **Create attention extraction hook** (`granite_attention_extractor.py`)
   ```python
   class AttentionExtractor:
       def __init__(self):
           self.attentions = []

       def hook_fn(self, module, input, output):
           # Extract cross-attention weights between audio and text tokens
           if hasattr(output, 'attentions'):
               self.attentions.append(output.attentions)

       def register_hooks(self, model):
           # Register hooks on all attention layers
           for layer in model.model.layers:
               layer.register_forward_hook(self.hook_fn)
   ```

2. **Enable attention output** in `GraniteTranscriber._transcribe_segment`
   ```python
   # Add output_attentions=True to generate() call
   generated_ids = self.model.generate(
       **inputs,
       max_new_tokens=500,
       output_attentions=True,  # NEW
       return_dict_in_generate=True  # NEW
   )
   ```

3. **Store attention matrices** for analysis
   - Save to numpy arrays or HDF5 format
   - Store alongside transcription for each audio file

**Validation**: Verify we can extract attention tensors shaped `[layers, heads, text_tokens, audio_tokens]`

---

### **PHASE 2: Analyze Attention Patterns & Find Monotonic Heads**

**Goal**: Identify which attention heads show temporal alignment patterns

**Files to Create**:
- `src/attention_analysis.py` (NEW)
- `scripts/analyze_granite_attentions.py` (NEW - standalone script)

**Implementation Steps**:

1. **Compute monotonicity score** for each attention head
   ```python
   def compute_monotonicity_score(attention_matrix):
       """
       Score how much later text tokens attend to later audio tokens.
       Higher score = stronger temporal alignment pattern.

       Args:
           attention_matrix: [text_tokens, audio_tokens]
       Returns:
           score: float
       """
       # For each text token position t, compute expected audio position
       # Check if E[audio_pos | text_pos=t] increases with t
       ...
   ```

2. **Rank all heads** across all layers
   - Process multiple audio samples
   - Average monotonicity scores
   - Select top 10-20 heads

3. **Visualize attention patterns** (optional but helpful)
   - Create heatmaps showing text_tokens (y) vs audio_tokens (x)
   - Save visualizations for manual inspection

**Validation**: Identify heads with scores > 0.8, similar to Layer 16 Head 23, Layer 34 Head 5, etc.

---

### **PHASE 3: Collect Training Data from LibriSpeech**

**Goal**: Generate training dataset with ground-truth word alignments

**Files to Create**:
- `scripts/prepare_librispeech_data.py` (NEW)
- `data/librispeech_alignments.csv` (NEW - generated data)

**Implementation Steps**:

1. **Download LibriSpeech corpus**
   ```bash
   # LibriSpeech test-clean subset (~300MB)
   wget http://www.openslr.org/resources/12/test-clean.tar.gz
   ```

2. **Extract word-level alignments**
   - LibriSpeech includes .txt transcripts
   - Use Montreal Forced Aligner or existing alignments to get word timings
   - Format: `{audio_file, word, start_time, end_time, word_index}`

3. **Run Granite inference with attention extraction**
   ```python
   for audio_file in librispeech_files:
       # Transcribe with Granite
       result, attentions = granite_transcribe_with_attentions(audio_file)

       # For each word in ground truth:
       for word, start_time in ground_truth:
           # Find corresponding token in Granite output
           token_idx = find_token_index(word, result['tokens'])

           # Extract attention values from top 10 heads
           features = extract_attention_features(attentions, token_idx, top_10_heads)

           # Store: (features, start_time) pair
           training_data.append((features, start_time))
   ```

4. **Save training dataset**
   - Format: NumPy arrays or PyTorch tensors
   - `X`: `[num_words, num_audio_tokens, 10_heads]`
   - `y`: `[num_words]` (start times in seconds)

**Validation**: Collect 10,000+ word-level training examples

---

### **PHASE 4: Train Auxiliary Alignment Model**

**Goal**: Train transformer model to predict word start times from attention patterns

**Files to Create**:
- `src/temporal_alignment_model.py` (NEW)
- `scripts/train_alignment_model.py` (NEW)
- `models/granite_temporal_aligner.pt` (NEW - saved weights)

**Implementation Steps**:

1. **Define model architecture**
   ```python
   class TemporalAlignmentModel(nn.Module):
       def __init__(self, num_heads=10, d_model=256, nhead=8, num_layers=4):
           super().__init__()
           # Project 10D input (one per attention head) to 256D
           self.input_proj = nn.Linear(num_heads, d_model)

           # Transformer encoder (4 layers, 8 heads, 256 dim, RoPE)
           self.transformer = nn.TransformerEncoder(
               nn.TransformerEncoderLayer(d_model, nhead),
               num_layers=num_layers
           )

           # Project to scalar logits (one per audio token)
           self.output_proj = nn.Linear(d_model, 1)

       def forward(self, x):
           # x: [batch, num_audio_tokens, 10]
           x = self.input_proj(x)  # [batch, num_audio_tokens, 256]
           x = self.transformer(x)
           logits = self.output_proj(x).squeeze(-1)  # [batch, num_audio_tokens]

           # Softmax over audio tokens
           weights = F.softmax(logits, dim=-1)

           # Compute weighted average of audio token positions
           positions = torch.arange(logits.size(1)) * 0.1  # 0.1s stride
           predicted_time = (weights * positions).sum(dim=-1)

           return predicted_time
   ```

2. **Training loop**
   ```python
   model = TemporalAlignmentModel()
   optimizer = Adam(model.parameters(), lr=1e-4)
   criterion = nn.MSELoss()

   for epoch in range(50):
       for batch_features, batch_times in train_loader:
           pred_times = model(batch_features)
           loss = criterion(pred_times, batch_times)

           optimizer.zero_grad()
           loss.backward()
           optimizer.step()
   ```

3. **Save trained model**
   ```python
   torch.save(model.state_dict(), 'models/granite_temporal_aligner.pt')
   ```

**Validation**: Achieve < 100ms mean absolute error on validation set

---

### **PHASE 5: Integrate into Production Pipeline**

**Goal**: Use trained model during Granite transcription

**Files to Modify**:
- `src/transcribe_audio.py` (GraniteTranscriber)

**Implementation Steps**:

1. **Load alignment model** in `GraniteTranscriber.__init__`
   ```python
   def __init__(self, model_name: str, device):
       super().__init__(model_name, device)
       self._load_model()

       # Load temporal alignment model
       self.alignment_model = TemporalAlignmentModel()
       self.alignment_model.load_state_dict(
           torch.load('models/granite_temporal_aligner.pt')
       )
       self.alignment_model.to(device)
       self.alignment_model.eval()
   ```

2. **Extract word-level timestamps** in `_transcribe_segment`
   ```python
   def _transcribe_segment(self, audio_path, start_time, end_time):
       # ... existing Granite transcription ...

       # NEW: Extract attention matrices
       generated_output = self.model.generate(
           **inputs,
           max_new_tokens=500,
           output_attentions=True,
           return_dict_in_generate=True
       )

       attentions = generated_output.attentions
       generated_ids = generated_output.sequences

       # Decode tokens
       transcription = self.processor.batch_decode(
           generated_ids, skip_special_tokens=True
       )[0]

       # Split into words
       words = transcription.split()

       # For each word, predict start time
       word_chunks = []
       for word_idx, word in enumerate(words):
           # Find token index for this word
           token_idx = find_word_token_index(word_idx, ...)

           # Extract attention features
           features = extract_features(attentions, token_idx, top_10_heads)

           # Predict start time
           with torch.no_grad():
               pred_start = self.alignment_model(features).item()

           # Create chunk
           word_chunks.append({
               "timestamp": (pred_start, pred_start + estimate_duration(word)),
               "text": word
           })

       return word_chunks
   ```

**Validation**: Test on real interview audio, compare with Whisper word timestamps

---

## Files Summary

### New Files to Create
1. **src/granite_attention_extractor.py** - Extract attention matrices
2. **src/attention_analysis.py** - Analyze & rank attention heads
3. **src/temporal_alignment_model.py** - Define alignment model architecture
4. **scripts/analyze_granite_attentions.py** - Standalone attention analysis script
5. **scripts/prepare_librispeech_data.py** - Generate training data
6. **scripts/train_alignment_model.py** - Train alignment model
7. **models/granite_temporal_aligner.pt** - Trained model weights

### Files to Modify
1. **src/transcribe_audio.py**
   - `GraniteTranscriber.__init__`: Load alignment model
   - `GraniteTranscriber._transcribe_segment`: Extract attentions & predict timestamps

## Dependencies

### Required (New)
```
# For dataset preparation
wget  # Download LibriSpeech
# OR use existing LibriSpeech if available

# For visualization (optional but recommended)
matplotlib>=3.5.0
seaborn>=0.12.0

# For model training
torch>=2.0.0  # Already have
transformers>=4.30.0  # Already have
```

### Optional (for ground truth alignment)
```
montreal-forced-aligner  # If LibriSpeech doesn't have alignments
# OR use existing aligned dataset
```

## Complexity & Timeline

### Effort Estimate
- **Phase 1** (Attention Extraction): 2-4 hours
- **Phase 2** (Head Analysis): 1-2 hours
- **Phase 3** (Data Collection): 4-8 hours (depends on dataset size)
- **Phase 4** (Model Training): 2-4 hours (training time depends on GPU)
- **Phase 5** (Integration): 2-3 hours

**Total**: 11-21 hours (spread over multiple days for training)

### Complexity Level
- **High**: Requires ML training, attention analysis, data preparation
- **Research-oriented**: May need experimentation to find best heads
- **GPU-intensive**: Training on HPC cluster recommended

### Risk Mitigation
- Start with Phase 1 to verify attention extraction works
- Validate head selection in Phase 2 before collecting full dataset
- Use small LibriSpeech subset initially (100 samples) for prototyping
- Full training only after validating approach

## Success Criteria

✅ **Phase 1**: Successfully extract attention tensors shaped `[40 layers, 40 heads, text_len, audio_len]`
✅ **Phase 2**: Identify 10+ heads with monotonicity score > 0.7
✅ **Phase 3**: Collect 10,000+ word-level training examples from LibriSpeech
✅ **Phase 4**: Train model with < 100ms mean absolute error on validation set
✅ **Phase 5**: Granite outputs word-level timestamps with ~90%+ accuracy

## Advantages Over Forced Alignment

1. **Granite-specific**: Uses model's internal knowledge, not external alignment
2. **Higher accuracy**: Should match or exceed 90% (vs 85-90% for CTC methods)
3. **Research contribution**: Novel approach, could publish/share results
4. **Interpretable**: Can visualize which heads learn temporal alignment

## Next Steps

**Before starting implementation**, we should:

1. ✅ **Verify attention extraction** works with Granite's API
   - Test `output_attentions=True` parameter
   - Check attention tensor shapes

2. **Download small LibriSpeech subset** for prototyping
   - Start with test-clean (~300MB, ~2600 utterances)
   - Verify we can access/use word-level alignments

3. **Decide on incremental approach**:
   - **Option A**: Implement all phases end-to-end
   - **Option B**: Start with Phase 1-2, validate, then continue

**Recommended**: Start with Phase 1 to validate technical feasibility before committing to full implementation.
