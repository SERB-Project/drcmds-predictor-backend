from fastapi import APIRouter, HTTPException, UploadFile, File, Form, Body
from app.services.sarsVariantsClassificationMutation.preprocess import preprocess_sequence as preprocess
from app.services.sarsVariantsClassificationMutation.predict import classify_variant as predict
from app.services.sarsVariantsClassificationMutation.parseFastaFiles import parse_fasta
from app.services.sarsVariantsClassificationMutation.nucleotideWiseMutations import explain_nucleotide_mutations
from app.services.sarsVariantsClassificationMutation.codonWiseMutations import explain_codon_mutations
from app.schemas.sarsPredictionReq import SequenceInput

router = APIRouter()

@router.post("/predictSarsSequence")
async def predict_sequence(sequence_input: SequenceInput):
    # Get sequence from JSON
    sequence = sequence_input.sequence
    
    # Preprocess and predict
    encoded_sequence = preprocess(sequence)
    prediction = predict(encoded_sequence)
    mutaions = explain_nucleotide_mutations(encoded_sequence)
    codon_wise_mutations = explain_codon_mutations(encoded_sequence)
    return {"variant": prediction, "mutations": mutaions, "codon_wise_mutations": codon_wise_mutations}

@router.post("/predictSarsFile")
async def predict_file(file: UploadFile):
    try:
        content = await file.read()
        sequence = parse_fasta(content.decode("utf-8"))
        
        # Preprocess and predict
        encoded_sequence = preprocess(sequence)
        prediction = predict(encoded_sequence)
        mutations = explain_nucleotide_mutations(encoded_sequence)
        codon_wise_mutations = explain_codon_mutations(encoded_sequence)

        return {"variant": prediction, "mutations": mutations, "codon_wise_mutations": codon_wise_mutations}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing FASTA file: {str(e)}")
