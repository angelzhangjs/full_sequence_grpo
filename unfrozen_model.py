
    # # ========================================================================
    # # Traditional Method - Direct unfreezing of attention layers
    # # ========================================================================
    # print("\n" + "="*70)
    # print("TRADITIONAL UNFREEZING (Attention Layers)")
    # print("="*70 + "\n")
    
    # # Freeze all parameters first
    # for param in model.parameters():
    #     param.requires_grad = False
    
    # # Configuration
    # TOTAL_BLOCKS = 28  # LTX-Video has 28 transformer blocks (0-27)
    # ATTN1_NUM_BLOCKS = 4  # Unfreeze self-attention in last 4 blocks
    # ATTN2_TARGET_BLOCK = 27  # Unfreeze cross-attention in last block only
    
    # attn1_start_block = TOTAL_BLOCKS - ATTN1_NUM_BLOCKS  # Block 23
    
    # unfrozen_params = []
    # attn1_count = 0
    # attn2_count = 0
    
    # print(f"🎯 Unfreezing Strategy:")
    # print(f"   Self-Attention (attn1): Blocks {attn1_start_block}-{TOTAL_BLOCKS-1} (last {ATTN1_NUM_BLOCKS} blocks)")
    # print(f"   Cross-Attention (attn2): Block {ATTN2_TARGET_BLOCK} only\n")
    
    # for name, param in model.named_parameters():
    #     should_unfreeze = False
    #     layer_type = None
        
    #     # Strategy 1: Unfreeze self-attention (attn1) in blocks 23-27
    #     for block_idx in range(attn1_start_block, TOTAL_BLOCKS):
    #         if f"transformer_blocks.{block_idx}." in name:
    #             # Look for self-attention patterns (based on actual layer names)
    #             if any(pattern in name for pattern in [
    #                 "attn1.to_q", "attn1.to_k", "attn1.to_v", "attn1.to_out"
    #             ]):
    #                 should_unfreeze = True
    #                 layer_type = "attn1"
    #                 attn1_count += 1
    #                 break
        
    #     # Strategy 2: Unfreeze cross-attention (attn2) ONLY in block 27
    #     if f"transformer_blocks.{ATTN2_TARGET_BLOCK}." in name:
    #         if any(pattern in name for pattern in [
    #             "attn2.to_q", "attn2.to_k", "attn2.to_v", "attn2.to_out"
    #         ]):
    #             should_unfreeze = True
    #             layer_type = "attn2"
    #             attn2_count += 1
        
    #     if should_unfreeze:
    #         param.requires_grad = True
    #         unfrozen_params.append(param)
    #         print(f"  Unfreezing [{layer_type:5s}]: {name} - {param.shape}")
    
    # # Safety check - raise error if attention layers not found
    # if len(unfrozen_params) == 0:
    #     print("\n❌ ERROR: No attention layers found!")
    #     print("\n🔍 Available layer names (sample):")
    #     sample_count = 0
    #     for name, _ in model.named_parameters():
    #         if 'block' in name.lower() and sample_count < 10:
    #             print(f"     {name}")
    #             sample_count += 1
        
    #     raise ValueError(
    #         f"No attention parameters were unfrozen!\n"
    #         f"   Looking for: Self-attention (attn1) in blocks {attn1_start_block}-{TOTAL_BLOCKS-1}\n"
    #         f"                Cross-attention (attn2) in block {ATTN2_TARGET_BLOCK}\n"
    #         f"   Check layer naming patterns above and adjust the code."
    #     )
    
    # # Summary
    # print(f"\n✅ Unfreezing Summary:")
    # print(f"   Self-attention (attn1) params: {attn1_count}")
    # print(f"   Cross-attention (attn2) params: {attn2_count}")
    # print(f"   Total unfrozen parameters: {len(unfrozen_params)}")
    # print(f"   Total param count: {sum(p.numel() for p in unfrozen_params):,}")
    
    # # Adjust learning rate based on number of parameters
    # total_params = sum(p.numel() for p in unfrozen_params)
    # if total_params > 10_000_000:  # > 10M params
    #     lr = 1e-5  # VERY gentle LR to preserve baseline quality!
    #     print(f"   Using GENTLE LR to preserve baseline: {lr:.2e}")
    # elif total_params > 1_000_000:  # > 1M params
    #     lr = 1e-4
    #     print(f"   Using moderate LR: {lr:.2e}")
    # else:
    #     lr = 1e-4
    #     print(f"   Using standard LR: {lr:.2e}")
    
    # print()
    
    # optimizer = torch.optim.Adam(
    #     unfrozen_params,
    #     lr=lr,
    #     betas=(0.9, 0.95),
    #     weight_decay=0.01
    # )
    