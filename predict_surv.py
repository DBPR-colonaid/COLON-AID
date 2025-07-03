import json
import argparse
import os.path
from utils import load_data, to_device, logits_to_risk, compute_risk_percentile
from modeling import build_model


def main():
    parser = argparse.ArgumentParser(description="Predict survival using a pre-trained model.")
    parser.add_argument('patient_data', type=str, default='data/samples/3.json',
                        help="Path to the patient data for prediction.")
    parser.add_argument('--model', type=str, default="model.pth", help="Path to the pre-trained model file.")

    args = parser.parse_args()

    model_path = args.model
    patient_data_path = args.patient_data

    # Load patient data
    traj, date_origin, info = load_data(patient_data_path)

    traj = to_device(traj, device='cpu')  # Move trajectory data to GPU if available

    # Load the pre-trained model
    model = build_model(model_path)

    model.cpu()
    model.eval()  # Set the model to evaluation mode

    output = model(traj, date_origin)

    predicted_risk = logits_to_risk(output['survival'])

    predicted_risk = predicted_risk.cpu().item()
    risk_percentile = compute_risk_percentile(predicted_risk)

    info_str = json.dumps(info, ensure_ascii=False, indent=4)
    print(f'Patient records: {info_str}')
    print(f'Predicted risk: {predicted_risk: .2f}\n'
          f'Risk Percentile: {risk_percentile: .2f}%')

    output = \
    f"""Patient records: {info_str}
    
    Predicted risk: {predicted_risk: .2f}
    Risk Percentile: {risk_percentile: .2f}%
    """
    pid = os.path.basename(patient_data_path).split('.')[0]  # Extract patient ID from the file name
    with open(f'output/{pid}.txt', 'w', encoding='utf-8') as f:
        f.write(output)


if __name__ == "__main__":
    main()
