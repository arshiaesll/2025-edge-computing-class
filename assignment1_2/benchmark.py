import torch
import time
import pandas as pd
from MiniVGG import ImageDataset, VGG
import os
import torch.quantization
import torch.nn.utils.prune as prune

def apply_pruning(model, amount=0.3):
    """Apply PyTorch's built-in pruning"""
    for name, module in model.named_modules():
        # Prune 30% of connections in all Conv2d layers
        if isinstance(module, torch.nn.Conv2d):
            prune.l1_unstructured(module, name='weight', amount=amount)
            # Make pruning permanent
            prune.remove(module, 'weight')
        # Prune 30% of connections in Linear layers
        elif isinstance(module, torch.nn.Linear):
            prune.l1_unstructured(module, name='weight', amount=amount)
            # Make pruning permanent
            prune.remove(module, 'weight')
    return model

def load_best_model():
    # Read summary.txt to find the best model
    with open('./summary.txt', 'r') as f:
        lines = f.readlines()
    
    # Parse model names and accuracies
    models = [(line.split(',')[0], float(line.split(',')[1])) for line in lines]
    # Get the model with highest accuracy
    best_model_name, best_accuracy = max(models, key=lambda x: x[1])
    print(f"Loading best model: {best_model_name} (accuracy: {best_accuracy:.4f})")
    expected_acc = best_accuracy
    
    # Load the model
    model = VGG(input_channels=3, num_classes=10)
    model.load_state_dict(torch.load(os.path.join('models', best_model_name), weights_only=True))
    model.eval()
    
    # Apply pruning
    # model = apply_pruning(model, amount=0.3)  # Prune 30% of weights
    print("Model pruned using PyTorch pruning!")
    
    # Convert to FP16
    model = model.half()
    print("Converted to FP16!")
    
    # Set device
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
        # Optimize model for inference using TorchScript
        try:
            print("Optimizing model for MPS...")
            model = model.to(device)
            # Create example input for tracing
            example_input = torch.randn(1, 3, 32, 32, dtype=torch.float16).to(device)
            # Script and optimize the model
            model = torch.jit.optimize_for_inference(
                torch.jit.script(model)
            )
            torch.backends.mps.graph_executor_enabled = True
            print("Model optimized successfully!")
        except Exception as e:
            print(f"Optimization failed, using standard model: {str(e)}")
    else:
        device = torch.device('cpu')
        model = model.to(device)
    # model = torch.compile(model, mode="reduce-overhead")

    return model, device, expected_acc

def run_benchmark():
    # Load model
    print("Loading best model...")
    model, device, expected_acc = load_best_model()
    
    # Load test dataset
    test_dataset = ImageDataset('./competition_data/test.csv', test=True)
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=1,  # Process one sample at a time to measure latency
        shuffle=False
    )
    
    # Warm-up phase
    print("\nWarming up...")
    dummy_input = torch.randn(1, 3, 32, 32, dtype=torch.float16).to(device)
    with torch.no_grad():
            for _ in range(5):  # Run 100 warm-up inferences
                _ = model(dummy_input)
    print("Warm-up complete!")
    
    results = []
    print("\nRunning inference on test dataset...")
    # with torch.no_grad():  # Disable gradient computation for inference
    with torch.inference_mode(mode = True):
        for i, (image, _) in enumerate(test_dataloader):
            image = image.half().to(device)  # Convert to FP16
            
            # Measure inference time
            start_time = time.time()
            output = model(image)
            end_time = time.time()
            
            # Get prediction and latency
            prediction = torch.argmax(output, dim=1).item()
            latency = (end_time - start_time) * 1000  # Convert to milliseconds
            
            results.append({
                'id': i,
                'label': prediction,
                'latency': latency
            })
            
            if (i + 1) % 100 == 0:
                print(f"Processed {i + 1} samples...")
    
    # Save results
    df = pd.DataFrame(results)
    output_file = 'submission.csv'
    df.to_csv(output_file, index=False)
    print(f"Results saved to {output_file}")
    
    # Print summary statistics
    print("\nBenchmark Summary:")
    print(f"Average latency: {df['latency'].mean():.2f} ms")
    print(f"Min latency: {df['latency'].min():.2f} ms")
    print(f"Max latency: {df['latency'].max():.2f} ms")
    print(f"Expected Score: {expected_acc / df['latency'].mean()}")

if __name__ == "__main__":
    run_benchmark()
