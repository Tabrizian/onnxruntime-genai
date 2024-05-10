#include "../generators.h"
#include "multi_modal_model.h"

namespace Generators {

MultiModalVisionModel::MultiModalVisionModel(std::unique_ptr<Config> config, OrtEnv& ort_env)
    : Model{std::move(config)} {
  embedding_session_ = OrtSession::Create(
      ort_env, (config_->config_path / config_->model.embeddings.filename).c_str(), session_options_.get());
  vision_session_ = OrtSession::Create(
      ort_env, (config_->config_path / config_->model.vision.filename).c_str(), session_options_.get());
  decoder_session_ = OrtSession::Create(
      ort_env, (config_->config_path / config_->model.decoder.filename).c_str(), session_options_.get());

  InitDeviceAllocator(*decoder_session_);
}

std::unique_ptr<State> MultiModalVisionModel::CreateState(RoamingArray<int32_t> sequence_lengths, const GeneratorParams& params) const {
  return std::make_unique<MultiModalPipelineState>(*this, sequence_lengths, params);
}

EmbeddingState::EmbeddingState(const MultiModalVisionModel& model, RoamingArray<int32_t> sequence_lengths, const GeneratorParams& params)
    : State{params},
      model_{model} {
  input_ids_.Add();
  embeddings_.Add();
}

// void InputEmbeddingsState::Run() {
//   State::Run(*model_.input_embeddings_session_, *model_.run_options_);
// }

VisionState::VisionState(const MultiModalVisionModel& model, RoamingArray<int32_t> sequence_lengths, const GeneratorParams& params)
    : State{params},
      model_{model} {
  input_embeddings_.Add();
  image_embeddings_.Add();
}

// void ImageProcessingState::Run() {
//   State::Run(*model_.input_embeddings_session_, *model_.run_options_);
// }

DecoderState::DecoderState(const MultiModalVisionModel& model, RoamingArray<int32_t> sequence_lengths, const GeneratorParams& params)
    : State{params},
      model_{model},
      position_inputs_{model, *this, sequence_lengths} {
  input_embeddings_.Add();
  position_inputs_.Add();
  logits_.Add();
  kv_cache_.Add();
}

// RoamingArray<float> MultiModalState::Run(int current_length, RoamingArray<int32_t> next_tokens, RoamingArray<int32_t> next_indices) {
//   if (first_run_) {
//     if (params_->use_cuda_graph) {
//       model_.run_options_->AddConfigEntry("gpu_graph_id", "-1");
//     }
//     first_run_ = false;
//   } else {
//     UpdateInputs(next_tokens, next_indices, current_length);
//   }

//   State::Run(*model_.session_decoder_, *model_.run_options_);

//   // Set the graph id for the following runs.
//   if (params_->use_cuda_graph) {
//     int new_batch_size = static_cast<int>(input_ids_.GetShape()[0]);
//     if (new_batch_size != current_batch_size_) {
//       current_batch_size_ = new_batch_size;
//       auto annotation_id = std::to_string(captured_graph_info_->GenerateUniqueAnnotationID(new_batch_size));
//       model_.run_options_->AddConfigEntry("gpu_graph_id", annotation_id.c_str());
//     }
//   }
//   return logits_.Get();
// }

void DecoderState::UpdateInputs(int current_length, RoamingArray<int32_t> beam_indices) {
  position_inputs_.Update(current_length);
  kv_cache_.Update(beam_indices.GetCPU(), current_length);
}

MultiModalPipelineState::MultiModalPipelineState(const MultiModalVisionModel& model,
                                                 RoamingArray<int32_t> sequence_lengths_unk,
                                                 const GeneratorParams& params)
    : State{params},
      model_{model},
      embedding_state_{std::make_unique<EmbeddingState>(model_, sequence_lengths_unk, params)},
      vision_state_{std::make_unique<VisionState>(model_, sequence_lengths_unk, params)},
      decoder_state_{std::make_unique<DecoderState>(model_, sequence_lengths_unk, params)} {
}

RoamingArray<float> MultiModalPipelineState::Run(int current_length, RoamingArray<int32_t> next_tokens, RoamingArray<int32_t> next_indices) {
  if (first_run_) {
    // Run all the three states with TransferStates in between.
  } else {
    // embedding_state_->UpdateInputsAndOutputs(next_tokens);
    decoder_state_->UpdateInputs(current_length, next_indices);
    // Run the embedding and decoder states with TransgerStates in between.
  }

  return RoamingArray<float>();
}

}  // namespace Generators
