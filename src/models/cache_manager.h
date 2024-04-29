// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once

#include <stdint.h>
#include <memory>
#include <vector>
#include <list>

namespace Generators {

struct CacheOptions {
  CacheOptions(const int32_t num_layers, const std::optional<int32_t>& block_size,
               const int32_t num_kv_heads, const int32_t head_size,
               const ONNXTensorElementDataType dtype,
               const std::optional<int32_t>& num_blocks,
               const std::optional<float>& gpu_utilization_factor);

  int32_t num_layers_{};
  int32_t block_size_{16};
  int32_t num_kv_heads_{};
  int32_t head_size_{};
  ONNXTensorElementDataType dtype_{};
  int32_t num_blocks_{};
  float gpu_utilization_factor_{0.3f};
};

class PagedCacheManager {
 public:
  PagedCacheManager(const CacheOptions& cache_options,
                    Ort::Allocator* cpu_allocator,
                    Ort::Allocator* gpu_allocator);

  // Returns the K, V cache for the given layer_id.
  std::pair<OrtValue*, OrtValue*> Cache(size_t layer_id);

  // Shape: [num_tokens, max_num_blocks_per_sequence]
  // Assume the cache contains the blocks for sequences with ids [2, 5, 7]
  // Assume the block tables for given sequence ids are:
  // {
  //   2: [0, 1, 2],
  //   5: [3, 7, 9],
  //   7: [4, 5, 6, 8]
  // }
  // Invoking this function will return the block tables as:
  // [
  //   [0, 1, 2, -1],
  //   [3, 7, 9, -1],
  //   [4, 5, 6, 8]
  // ]
  //
  // This implies that the sequence at index 0 (sequence id 2) has its kv cache stored in blocks with ids [0, 1, 2],
  // the sequence at index 1 (sequence id 5) has its kv cache stored in blocks with ids [3, 7, 9], and and
  // the sequence at index 2 (sequence id 7) has its kv cache stored in blocks with ids [4, 5, 6, 8].
  // -1 is used to pad the block tables to the max blocks per sequence from the given sequences.
  // The order of the block tables is based on the order the sequences were added.
  OrtValue* BlockTables();

  // Shape: [num_tokens]
  // Prompt stage:
  // Assume the cache contains the blocks for sequences with ids [2, 5, 7]
  // Assume that the slot mapping for the given sequence ids are:
  // {
  //   2: 32, 33, 34, 35
  //   5: 0, 1, 2, 3, 4
  //   7: 16, 17, 18
  // }
  // And assume that the block size is 16.
  // The slot mapping tells us that the sequence with id 2 should fill its prompt KV cache tokens at slots
  // [0, 1, 2, 3] (slot_id % 16) in block 2 (slot_id / 16), the sequence with id 5 should fill its prompt KV
  // cache tokens at slots [0, 1, 2, 3, 4] in block 0, and the sequence with id 7 should fill its prompt KV
  // cache tokens at slots [0, 1, 2] in block 1.
  // Invoking this function will return the slot mapping as:
  // [ | 32, 33, 34, 35, | 0, 1, 2, 3, 4, | 16, 17, 18 | ]
  // Decoding stage:
  // The same principle applies for the decoding stage, but the slot mapping will only contain
  // the slot ids for the new token generated by the model.
  // For example, assume that the cache contains the blocks for sequences with ids [2, 5, 7]
  // Assume that the slot mapping for the given sequence ids are:
  // {
  //   2: 43,
  //   5: 29,
  //   7: 12
  // }
  // And assume that the block size is 16.
  // The slot mapping tells us that the sequence with id 2 should fill its KV cache token at slot
  // 11 (43 % 16) in block 2 (43 / 16), the sequence with id 5 should fill its KV cache token at slot
  // 13 (29 % 16) in block 1, and the sequence with id 7 should fill its KV cache token at slot
  // 12 (12 % 16) in block 0.
  // The order of the slot mapping is based on the order the sequences were added.
  OrtValue* SlotMapping();

  // Removes the allocated blocks for the given sequence_id and makes it available for
  // other sequences.
  void Remove(size_t sequence_id);

  // Allocates blocks needed to serve the given sequence_id for the given prompt token size.
  // Cache additions happen one sequence at a time.
  void Add(size_t sequence_id, size_t prompt_token_size);

  // Before running a decoding step, the cache needs to allot a new slot for the given sequence_id.
  // If the block has been completely filled up, a new block will be allocated as well.
  // This function should be called before each decoding step.
  void AddToken(size_t sequence_id);

  // Reorder the cache based on the given order.
  // This is needed when the order of the inputs changes due to beam search.
  void ReorderCache(const std::vector<size_t>& index_permutation);

  // Returns the order of the sequences in the cache.
  // std::vector<size_t> Order() const;

 private:
  using LayerCache = std::unique_ptr<OrtValue>;  // Shape: [num_blocks, block_size * num_kv_heads * head_size]
  /*
  The K and the V Cache is represented as an array of blocks. Each block contains
  a number of slots equal to the block size. Each slot contains num_kv_heads * head_size
  elements. Here the slot represents data generated by the model for a single token.
  This KV cache is allocated for each layer in the model.
  Although the cache is preallocated, the actual memory is alloted to a sequence_id only as needed.

  View of the cache for each layer (LayerCache):

        -->|size of each block = block_size(M) * size of each slot|<--
           |______________________________________________________|
           |       -->|          |<-- size of each slot = num_kv_heads * head_size
           |          |          |                                |
           |__________|__________|________________________________|
  block 0  |  slot 0  |  slot 1  |  slot 2  |     .    |  slot M  |
  block 1  |          |          |          |          |          |
  block 2  |          |          |          |          |          |
  block 3  |          |          |          |          |          |
     .     |          |          |          |          |          |
     .     |          |          |          |          |          |
     .     |          |          |          |          |          |
           |          |          |          |          |          |
  block N  |__________|__________|__________|__________|__________|

  N = num_blocks per layer
  M = block_size per block

  */

  struct BlockInfoPerSequence {
    size_t sequence_id;
    bool is_prompt;
    std::vector<size_t> block_ids;  // List of block_ids alloted to the sequence_id
    std::vector<size_t> slot_ids;   // Slot id of the slot to use for the input token
    size_t context_length;          // Context length of the sequence.
                                    // = prompt_tokens.size() for prompt stage
                                    // = prompt_tokens.size() + generated_tokens() for decoding stage
  };

  std::vector<size_t> FindAvailableBlocks(size_t num_blocks);
  void ReserveBlocks(const std::vector<size_t>& block_ids);
  void ReleaseBlocks(const std::vector<size_t>& block_ids);

  CacheOptions options_;
  Ort::Allocator* cpu_allocator_;
  Ort::Allocator* gpu_allocator_;
  std::vector<std::pair<LayerCache, LayerCache>> cache_;                                // Pair of key and value caches for all layers
  std::vector<int32_t> block_refs_;                                                     // List of free blocks
  std::list<BlockInfoPerSequence> block_infos_;                                         // List of block_info for all sequences
  std::unordered_map<size_t, std::list<BlockInfoPerSequence>::iterator> block_tables_;  // Mapping of sequence_id to block_info
  std::unique_ptr<OrtValue> block_tables_value_;
  std::unique_ptr<OrtValue> slot_mapping_value_;
};

}  // namespace Generators
