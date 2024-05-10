// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "generators.h"
#include "models/model.h"

#include <regex>

namespace Generators {

namespace {

}  // namespace

std::vector<int32_t> ProcessImagePrompt(const Generators::Tokenizer& tokenizer, const std::string& prompt,
                                        const size_t num_patches, const size_t num_image_tokens) {
  const std::regex pattern("<\\|image_\\d+\\|>");
  const std::vector<std::string> prompt_chunks(
      std::sregex_token_iterator(prompt.begin(), prompt.end(), pattern, -1),
      std::sregex_token_iterator());

  std::vector<std::vector<int32_t>> input_ids_chunks(prompt_chunks.size());
  for (size_t i = 0; i < prompt_chunks.size(); ++i) {
    input_ids_chunks[i] = tokenizer.Encode(prompt_chunks[i].c_str());
  }

  const std::vector<std::string> image_tags(
      std::sregex_token_iterator(prompt.begin(), prompt.end(), pattern),
      std::sregex_token_iterator());

  std::vector<int32_t> image_ids(image_tags.size());
  constexpr size_t image_id_position_begin = 9;  // <\|image_ -> 9 characters
  for (size_t i = 0; i < image_tags.size(); ++i) {
    const size_t image_id_position_end = image_tags[i].size() - 1;  // |> -> 1 character to the end '|'
    image_ids[i] = std::stoi(image_tags[i].substr(image_id_position_begin,
                                                  image_id_position_end - image_id_position_begin));
  }

  std::vector<int32_t> input_ids;
  for (size_t i = 0; i < input_ids_chunks.size(); ++i) {
    input_ids.insert(input_ids.end(), input_ids_chunks[i].begin(), input_ids_chunks[i].end());
    if (i < image_ids.size()) {
      for (size_t j = 0; j < num_patches * num_image_tokens; ++j) {
        input_ids.push_back(-image_ids[i]);
      }
    }
  }

  return input_ids;
}

}  // namespace Generators
