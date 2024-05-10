// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "model.h"
#include "input_ids.h"
#include "input_embeds.h"
#include "logits.h"
#include "kv_cache.h"
#include "position_inputs.h"

namespace Generators {

class MultiModalVisionModel : public Model {
 public:
  MultiModalVisionModel(std::unique_ptr<Config> config, OrtEnv& ort_env);

  std::unique_ptr<State> CreateState(RoamingArray<int32_t> sequence_lengths, const GeneratorParams& params) const override;

 private:
  std::unique_ptr<OrtSession> embedding_session_;
  std::unique_ptr<OrtSession> vision_session_;
  std::unique_ptr<OrtSession> decoder_session_;
};

class EmbeddingState : public State {
 public:
  EmbeddingState(const MultiModalVisionModel& model, RoamingArray<int32_t> sequence_lengths, const GeneratorParams& params);
  RoamingArray<float> Run(int current_length, RoamingArray<int32_t> next_tokens, RoamingArray<int32_t> next_indices = {}) override { return RoamingArray<float>(); }

 private:
  const MultiModalVisionModel& model_;

  InputIDs input_ids_{model_, *this};                             // Model input
  Embeddings embeddings_{model_, this, Embeddings::Mode::Output,  // Model output
                         model_.config_->model.embeddings.outputs.embeddings};
};

class VisionState : public State {
 public:
  VisionState(const MultiModalVisionModel& model, RoamingArray<int32_t> sequence_lengths, const GeneratorParams& params);
  RoamingArray<float> Run(int current_length, RoamingArray<int32_t> next_tokens, RoamingArray<int32_t> next_indices = {}) override { return RoamingArray<float>(); }

 private:
  const MultiModalVisionModel& model_;

  Embeddings input_embeddings_{model_, this, Embeddings::Mode::Input,  // Model input
                               model_.config_->model.vision.inputs.embeddings};
  Embeddings image_embeddings_{model_, this, Embeddings::Mode::Output,  // Model output
                               model_.config_->model.vision.outputs.embeddings};
};

class DecoderState : public State {
 public:
  DecoderState(const MultiModalVisionModel& model, RoamingArray<int32_t> sequence_lengths, const GeneratorParams& params);
  RoamingArray<float> Run(int current_length, RoamingArray<int32_t> next_tokens, RoamingArray<int32_t> next_indices) override;
  const CapturedGraphInfo* GetCapturedGraphInfo() const override { return captured_graph_info_.get(); };

  void UpdateInputs(int current_length, RoamingArray<int32_t> beam_indices);

 private:
  const MultiModalVisionModel& model_;
  CapturedGraphInfoPtr captured_graph_info_;
  int current_batch_size_{};

  Embeddings input_embeddings_{model_, this, Embeddings::Mode::Input,  // Model output
                               model_.config_->model.decoder.inputs.embeddings};
  PositionInputs position_inputs_;    // Model input
  KV_Cache kv_cache_{model_, *this};  // Model input
  Logits logits_{model_, *this};      // Model output
};

class MultiModalPipelineState : public State {
 public:
  MultiModalPipelineState(const MultiModalVisionModel& model, RoamingArray<int32_t> sequence_lengths, const GeneratorParams& params);
  RoamingArray<float> Run(int current_length, RoamingArray<int32_t> next_tokens, RoamingArray<int32_t> next_indices) override;

 private:
  void UpdateInputs(const RoamingArray<int32_t>& next_tokens, RoamingArray<int32_t> next_indices, int current_length);

  const MultiModalVisionModel& model_;
  std::unique_ptr<EmbeddingState> embedding_state_;
  std::unique_ptr<VisionState> vision_state_;
  std::unique_ptr<DecoderState> decoder_state_;
  bool first_run_{true};
};

}  // namespace Generators
