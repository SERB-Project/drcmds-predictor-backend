from app.models.pathogenicityClassification.pathogenicityClassifying_model_load import model

def predict_pathogenicity(processed_df):
    prediction = model.predict(processed_df)[0]
    return "Pathogenic" if prediction == 1 else "Benign"