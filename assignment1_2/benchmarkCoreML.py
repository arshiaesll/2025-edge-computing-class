import torch
import time
import pandas as pd
from MiniVGG import ImageDataset, VGG
import os
import coremltools as ct
import numpy as np

def load_and_convert_model():
    # Read summary.txt to find the best model
    with open('./summary.txt', 'r') as f:
        lines = f.readlines()
    
    # Parse model names and accuracies
    models = [(line.split(',')[0], float(line.split(',')[1])) for line in lines]
    best_model_name, best_accuracy = max(models, key=lambda x: x[1])
    print(f"Loading best model: {best_model_name} (accuracy: {best_accuracy:.4f})")
    
    # Load PyTorch model
    model = VGG(input_channels=3, num_classes=10)
    model.load_state_dict(torch.load(os.path.join('models', best_model_name), weights_only=True))
    model.eval()
    
    # Convert to CoreML
    print("Converting to CoreML...")
    example_input = torch.randn(1, 3, 32, 32)
    traced_model = torch.jit.trace(model, example_input)
    
    coreml_model = ct.convert(
        traced_model,
        inputs=[ct.TensorType(name='x', shape=example_input.shape)],
        compute_units=ct.ComputeUnit.ALL  # Use all available compute units
    )
    
    # Save the CoreML model
    coreml_model.save("model.mlpackage")
    print("Model converted and saved as model.mlpackage")
    
    return coreml_model, best_accuracy

def run_benchmark():
    # Load and convert model
    print("Loading and converting model...")
    model, expected_acc = load_and_convert_model()
    
    # Load test dataset
    test_dataset = ImageDataset('./competition_data/test.csv', test=True)
    
    # Warm-up phase
    print("\nWarming up...")
    dummy_input = np.random.rand(1, 3, 32, 32).astype(np.float32)
    for _ in range(5):
        _ = model.predict({'x': dummy_input})
    print("Warm-up complete!")
    
    results = []
    print("\nRunning inference on test dataset...")
    
    for i in range(len(test_dataset)):
        image, _ = test_dataset[i]
        # Convert to numpy and add batch dimension
        image_np = image.numpy()[None, ...]
        
        # Measure inference time
        start_time = time.time()
        output = model.predict({'x': image_np})
        end_time = time.time()
        
        # Get prediction and latency
        prediction = np.argmax(output['var_104'])  # Using the renamed output
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
