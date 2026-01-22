from codebleu import calc_codebleu

def evaluate_model(predictions, references):
    """
    Calculates CodeBLEU score comparing model output to ground truth.
    """
    result = calc_codebleu(references, predictions, lang="python", weights=(0.25, 0.25, 0.25, 0.25))
    print(f"CodeBLEU Score: {result['codebleu']}")
    return result

# Example Usage
if __name__ == "__main__":
    ref = ["def add(a, b):\n    return a + b"]
    pred = ["def sum(x, y):\n    return x + y"]
    evaluate_model(pred, ref)