"""
Batch processing with visualizations for selected samples
"""

from standalone_ig_example import *

def batch_analysis_with_visualizations(unet_model, start_idx=0, end_idx=10, 
                                     steps=30, visualize_every=5):
    """
    Run batch analysis with visualizations for selected samples
    
    Args:
        unet_model: Trained model
        start_idx: Start index
        end_idx: End index  
        steps: Integration steps
        visualize_every: Create visualization every N samples
    """
    print("="*80)
    print("BATCH ANALYSIS WITH SELECTIVE VISUALIZATIONS")
    print("="*80)
    
    batch_results = []
    
    for idx in range(start_idx, end_idx):
        try:
            print(f"\nProcessing sample {idx}...")
            
            # Should we create visualization for this sample?
            create_viz = (idx % visualize_every == 0) or (idx == start_idx)
            
            if create_viz:
                # Full analysis with visualization
                results = run_integrated_gradients_analysis(
                    unet_model=unet_model,
                    image_index=idx,
                    steps=steps,
                    target_mode='mean_confidence'
                )
            else:
                # Fast analysis without visualization
                results = run_integrated_gradients_analysis_batch(
                    unet_model=unet_model,
                    image_index=idx,
                    steps=steps,
                    target_mode='mean_confidence'
                )
            
            batch_results.append(results)
            print(f"✓ Sample {idx} completed - Patient {results['sample_info']['patient_id']}")
            
        except Exception as e:
            print(f"❌ Error processing sample {idx}: {e}")
            continue
    
    print(f"\n✓ Batch completed! Processed {len(batch_results)} samples")
    print(f"✓ Created visualizations for samples: {list(range(start_idx, end_idx, visualize_every))}")
    
    return batch_results


def create_summary_visualization(batch_results, save_path=None):
    """
    Create a summary visualization showing multiple samples
    """
    n_samples = min(len(batch_results), 6)  # Show up to 6 samples
    
    fig, axes = plt.subplots(2, n_samples, figsize=(4*n_samples, 8))
    
    for i in range(n_samples):
        result = batch_results[i]
        
        # Original image
        image_2d = np.squeeze(result['image'])
        axes[0, i].imshow(image_2d, cmap='gray')
        axes[0, i].set_title(f"Patient {result['sample_info']['patient_id']}\n{result['sample_info']['class_name']}")
        axes[0, i].axis('off')
        
        # Integrated gradients
        ig_2d = np.squeeze(result['integrated_grads'])
        ig_enhanced = ig_2d * 1000
        axes[1, i].imshow(ig_enhanced, cmap='RdBu_r')
        axes[1, i].set_title('Integrated Gradients')
        axes[1, i].axis('off')
    
    plt.suptitle('Batch Integrated Gradients Summary', fontsize=16)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Summary visualization saved to: {save_path}")
    
    plt.show()


if __name__ == "__main__":
    # Example usage
    unet_model = load_unet_model()
    
    if unet_model:
        # Analyze first 20 samples with visualization every 5 samples
        results = batch_analysis_with_visualizations(
            unet_model=unet_model,
            start_idx=0,
            end_idx=20,
            steps=30,
            visualize_every=5  # Visualize samples 0, 5, 10, 15
        )
        
        # Create summary visualization
        create_summary_visualization(
            results, 
            save_path="workspace/output_latest/batch_summary_visualization.png"
        )