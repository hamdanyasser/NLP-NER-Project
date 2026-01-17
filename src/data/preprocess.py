"""
Data preprocessing module for Named Entity Recognition.

This module handles:
1. Processing the BC5CDR biomedical NER dataset
2. Generating high-quality sample data as fallback
3. Converting to BIO tagging format
4. Splitting into train/dev/test sets
5. Computing dataset statistics

Author: NLP Course Project
"""

import os
import random
from typing import List, Tuple, Dict, Set
from pathlib import Path
from collections import defaultdict

# Import BC5CDR parser
try:
    from src.data.bc5cdr_parser import (
        process_bc5cdr_dataset,
        get_dataset_statistics,
        print_dataset_statistics,
        validate_bio_sequence,
        fix_bio_sequence
    )
except ImportError:
    from bc5cdr_parser import (
        process_bc5cdr_dataset,
        get_dataset_statistics,
        print_dataset_statistics,
        validate_bio_sequence,
        fix_bio_sequence
    )


def create_comprehensive_sample_data() -> List[Tuple[List[str], List[str]]]:
    """
    Create a comprehensive sample biomedical NER dataset.

    This creates UNIQUE, varied sentences (no duplication) covering:
    - Various chemical compounds (drugs, molecules)
    - Various diseases (conditions, syndromes)
    - Different sentence structures
    - Multi-word entities

    Returns:
        List of (tokens, tags) tuples
    """
    # Sample data organized by category for diversity
    sample_sentences = []

    # Category 1: Drug-Disease relationships
    drug_disease_samples = [
        ("Aspirin is commonly used to treat headaches and reduce fever .",
         [("Aspirin", "Chemical"), ("headaches", "Disease"), ("fever", "Disease")]),

        ("Ibuprofen provides relief from arthritis pain and inflammation .",
         [("Ibuprofen", "Chemical"), ("arthritis", "Disease"), ("inflammation", "Disease")]),

        ("Metformin is the first-line treatment for type 2 diabetes .",
         [("Metformin", "Chemical"), ("type 2 diabetes", "Disease")]),

        ("Patients taking warfarin should avoid vitamin K rich foods .",
         [("warfarin", "Chemical"), ("vitamin K", "Chemical")]),

        ("Lisinopril effectively lowers blood pressure in hypertension .",
         [("Lisinopril", "Chemical"), ("hypertension", "Disease")]),

        ("Omeprazole treats gastroesophageal reflux disease and ulcers .",
         [("Omeprazole", "Chemical"), ("gastroesophageal reflux disease", "Disease"), ("ulcers", "Disease")]),

        ("Atorvastatin reduces cholesterol levels and cardiovascular risk .",
         [("Atorvastatin", "Chemical"), ("cholesterol", "Chemical")]),

        ("Amoxicillin is prescribed for bacterial pneumonia treatment .",
         [("Amoxicillin", "Chemical"), ("pneumonia", "Disease")]),

        ("Prednisone suppresses the immune response in autoimmune disorders .",
         [("Prednisone", "Chemical"), ("autoimmune disorders", "Disease")]),

        ("Gabapentin helps manage neuropathic pain and seizures .",
         [("Gabapentin", "Chemical"), ("neuropathic pain", "Disease"), ("seizures", "Disease")]),
    ]

    # Category 2: Side effects and adverse reactions
    adverse_effect_samples = [
        ("Acetaminophen overdose can cause severe hepatotoxicity .",
         [("Acetaminophen", "Chemical"), ("hepatotoxicity", "Disease")]),

        ("Cisplatin chemotherapy often results in nephrotoxicity .",
         [("Cisplatin", "Chemical"), ("nephrotoxicity", "Disease")]),

        ("Prolonged corticosteroid use leads to osteoporosis risk .",
         [("corticosteroid", "Chemical"), ("osteoporosis", "Disease")]),

        ("Statins may cause muscle pain known as myopathy .",
         [("Statins", "Chemical"), ("myopathy", "Disease")]),

        ("Penicillin allergies can trigger anaphylaxis in sensitive patients .",
         [("Penicillin", "Chemical"), ("anaphylaxis", "Disease")]),

        ("Opioid medications carry significant addiction risk .",
         [("Opioid", "Chemical"), ("addiction", "Disease")]),

        ("Methotrexate therapy requires monitoring for bone marrow suppression .",
         [("Methotrexate", "Chemical"), ("bone marrow suppression", "Disease")]),

        ("Vancomycin can cause red man syndrome during infusion .",
         [("Vancomycin", "Chemical"), ("red man syndrome", "Disease")]),

        ("Lithium toxicity presents with tremor and confusion .",
         [("Lithium", "Chemical"), ("tremor", "Disease"), ("confusion", "Disease")]),

        ("Amphotericin B is associated with infusion-related reactions .",
         [("Amphotericin B", "Chemical")]),
    ]

    # Category 3: Disease descriptions
    disease_samples = [
        ("Alzheimer disease causes progressive memory loss and dementia .",
         [("Alzheimer disease", "Disease"), ("memory loss", "Disease"), ("dementia", "Disease")]),

        ("Parkinson disease is characterized by tremor and bradykinesia .",
         [("Parkinson disease", "Disease"), ("tremor", "Disease"), ("bradykinesia", "Disease")]),

        ("Multiple sclerosis affects the central nervous system myelin .",
         [("Multiple sclerosis", "Disease")]),

        ("Rheumatoid arthritis causes joint inflammation and destruction .",
         [("Rheumatoid arthritis", "Disease"), ("inflammation", "Disease")]),

        ("Chronic obstructive pulmonary disease impairs breathing function .",
         [("Chronic obstructive pulmonary disease", "Disease")]),

        ("Heart failure results in fluid retention and dyspnea .",
         [("Heart failure", "Disease"), ("dyspnea", "Disease")]),

        ("Systemic lupus erythematosus is an autoimmune condition .",
         [("Systemic lupus erythematosus", "Disease")]),

        ("Crohn disease causes intestinal inflammation and pain .",
         [("Crohn disease", "Disease"), ("inflammation", "Disease")]),

        ("Migraine headaches are often preceded by visual aura .",
         [("Migraine", "Disease"), ("headaches", "Disease")]),

        ("Epilepsy is characterized by recurrent seizure episodes .",
         [("Epilepsy", "Disease"), ("seizure", "Disease")]),
    ]

    # Category 4: Treatment protocols
    treatment_samples = [
        ("Insulin therapy maintains glucose control in diabetes mellitus .",
         [("Insulin", "Chemical"), ("glucose", "Chemical"), ("diabetes mellitus", "Disease")]),

        ("Dopamine agonists are used in early Parkinson disease .",
         [("Dopamine", "Chemical"), ("Parkinson disease", "Disease")]),

        ("Interferon beta reduces relapse rates in multiple sclerosis .",
         [("Interferon beta", "Chemical"), ("multiple sclerosis", "Disease")]),

        ("TNF inhibitors revolutionized rheumatoid arthritis treatment .",
         [("TNF", "Chemical"), ("rheumatoid arthritis", "Disease")]),

        ("Epinephrine is the first-line treatment for anaphylaxis .",
         [("Epinephrine", "Chemical"), ("anaphylaxis", "Disease")]),

        ("Nitroglycerin provides rapid relief in angina pectoris .",
         [("Nitroglycerin", "Chemical"), ("angina pectoris", "Disease")]),

        ("Albuterol inhalers quickly relieve acute asthma symptoms .",
         [("Albuterol", "Chemical"), ("asthma", "Disease")]),

        ("Levodopa remains the gold standard for Parkinson disease .",
         [("Levodopa", "Chemical"), ("Parkinson disease", "Disease")]),

        ("Morphine is essential for severe cancer pain management .",
         [("Morphine", "Chemical"), ("cancer", "Disease")]),

        ("Heparin prevents thrombosis in hospitalized patients .",
         [("Heparin", "Chemical"), ("thrombosis", "Disease")]),
    ]

    # Category 5: Clinical findings
    clinical_samples = [
        ("The patient presented with fever and elevated white blood cells .",
         [("fever", "Disease")]),

        ("Laboratory tests revealed hypoglycemia and ketoacidosis .",
         [("hypoglycemia", "Disease"), ("ketoacidosis", "Disease")]),

        ("Chest X-ray confirmed bilateral pneumonia infiltrates .",
         [("pneumonia", "Disease")]),

        ("ECG showed atrial fibrillation with rapid ventricular response .",
         [("atrial fibrillation", "Disease")]),

        ("MRI revealed multiple brain lesions consistent with stroke .",
         [("stroke", "Disease")]),

        ("Blood cultures grew Staphylococcus aureus bacteremia .",
         [("bacteremia", "Disease")]),

        ("Echocardiogram demonstrated severe mitral regurgitation .",
         [("mitral regurgitation", "Disease")]),

        ("CT scan identified acute appendicitis requiring surgery .",
         [("appendicitis", "Disease")]),

        ("Lumbar puncture showed elevated protein suggesting meningitis .",
         [("meningitis", "Disease")]),

        ("Biopsy confirmed adenocarcinoma of the lung tissue .",
         [("adenocarcinoma", "Disease")]),
    ]

    # Category 6: Biochemistry and mechanisms
    mechanism_samples = [
        ("Serotonin reuptake inhibitors increase synaptic serotonin .",
         [("Serotonin", "Chemical"), ("serotonin", "Chemical")]),

        ("ACE inhibitors block angiotensin converting enzyme activity .",
         [("angiotensin", "Chemical")]),

        ("Beta blockers reduce heart rate by blocking catecholamine effects .",
         [("catecholamine", "Chemical")]),

        ("Calcium channel blockers prevent calcium influx in cardiac cells .",
         [("Calcium", "Chemical"), ("calcium", "Chemical")]),

        ("Proton pump inhibitors reduce gastric acid secretion .",
         [("acid", "Chemical")]),

        ("Antihistamines block histamine receptors reducing allergic symptoms .",
         [("histamine", "Chemical")]),

        ("Thrombolytics dissolve fibrin clots in acute myocardial infarction .",
         [("fibrin", "Chemical"), ("myocardial infarction", "Disease")]),

        ("Benzodiazepines enhance GABA receptor activity in anxiety .",
         [("Benzodiazepines", "Chemical"), ("GABA", "Chemical"), ("anxiety", "Disease")]),

        ("Antipsychotics block dopamine D2 receptors in schizophrenia .",
         [("dopamine", "Chemical"), ("schizophrenia", "Disease")]),

        ("NSAIDs inhibit cyclooxygenase reducing prostaglandin synthesis .",
         [("NSAIDs", "Chemical"), ("cyclooxygenase", "Chemical"), ("prostaglandin", "Chemical")]),
    ]

    # Category 7: Additional varied sentences
    varied_samples = [
        ("Vitamin D supplementation prevents rickets in children .",
         [("Vitamin D", "Chemical"), ("rickets", "Disease")]),

        ("Iron deficiency leads to microcytic anemia symptoms .",
         [("Iron", "Chemical"), ("anemia", "Disease")]),

        ("Thyroid hormone replacement treats hypothyroidism effectively .",
         [("Thyroid hormone", "Chemical"), ("hypothyroidism", "Disease")]),

        ("Potassium chloride corrects hypokalemia in hospitalized patients .",
         [("Potassium chloride", "Chemical"), ("hypokalemia", "Disease")]),

        ("Magnesium sulfate prevents seizures in preeclampsia patients .",
         [("Magnesium sulfate", "Chemical"), ("seizures", "Disease"), ("preeclampsia", "Disease")]),

        ("Sodium bicarbonate treats metabolic acidosis in critical care .",
         [("Sodium bicarbonate", "Chemical"), ("metabolic acidosis", "Disease")]),

        ("Folic acid supplementation prevents neural tube defects .",
         [("Folic acid", "Chemical"), ("neural tube defects", "Disease")]),

        ("Zinc lozenges may reduce common cold duration slightly .",
         [("Zinc", "Chemical"), ("common cold", "Disease")]),

        ("Activated charcoal absorbs toxins in poisoning cases .",
         [("charcoal", "Chemical")]),

        ("Naloxone rapidly reverses opioid overdose effects .",
         [("Naloxone", "Chemical"), ("overdose", "Disease")]),
    ]

    # Category 8: Complex multi-entity sentences
    complex_samples = [
        ("Combination therapy with aspirin and clopidogrel prevents stroke recurrence .",
         [("aspirin", "Chemical"), ("clopidogrel", "Chemical"), ("stroke", "Disease")]),

        ("Triple therapy includes omeprazole amoxicillin and clarithromycin for H. pylori .",
         [("omeprazole", "Chemical"), ("amoxicillin", "Chemical"), ("clarithromycin", "Chemical")]),

        ("HAART combines multiple antiretroviral drugs for HIV treatment .",
         [("antiretroviral", "Chemical"), ("HIV", "Disease")]),

        ("Chemotherapy regimens often include cyclophosphamide and doxorubicin .",
         [("cyclophosphamide", "Chemical"), ("doxorubicin", "Chemical")]),

        ("Immunosuppression with tacrolimus and mycophenolate prevents transplant rejection .",
         [("tacrolimus", "Chemical"), ("mycophenolate", "Chemical")]),

        ("Combination inhaler contains fluticasone and salmeterol for asthma .",
         [("fluticasone", "Chemical"), ("salmeterol", "Chemical"), ("asthma", "Disease")]),

        ("Dual antiplatelet therapy reduces coronary artery disease events .",
         [("coronary artery disease", "Disease")]),

        ("Treatment of tuberculosis requires rifampin isoniazid and pyrazinamide .",
         [("tuberculosis", "Disease"), ("rifampin", "Chemical"), ("isoniazid", "Chemical"), ("pyrazinamide", "Chemical")]),

        ("Sepsis management includes antibiotics fluids and vasopressors .",
         [("Sepsis", "Disease"), ("antibiotics", "Chemical"), ("vasopressors", "Chemical")]),

        ("Cancer immunotherapy uses pembrolizumab and nivolumab checkpoint inhibitors .",
         [("Cancer", "Disease"), ("pembrolizumab", "Chemical"), ("nivolumab", "Chemical")]),
    ]

    # Combine all samples
    all_samples = (
        drug_disease_samples +
        adverse_effect_samples +
        disease_samples +
        treatment_samples +
        clinical_samples +
        mechanism_samples +
        varied_samples +
        complex_samples
    )

    # Convert to BIO format
    bio_data = []
    for sentence, entities in all_samples:
        tokens = sentence.split()
        tags = ['O'] * len(tokens)

        # Create entity lookup for multi-word matching
        for entity_text, entity_type in entities:
            entity_tokens = entity_text.split()
            entity_len = len(entity_tokens)

            # Find entity in tokens
            for i in range(len(tokens) - entity_len + 1):
                # Check if tokens match entity (case-insensitive)
                if [t.lower().rstrip('.,;:!?') for t in tokens[i:i+entity_len]] == \
                   [t.lower() for t in entity_tokens]:
                    # Only tag if not already tagged
                    if tags[i] == 'O':
                        tags[i] = f'B-{entity_type}'
                        for j in range(1, entity_len):
                            if i + j < len(tags):
                                tags[i + j] = f'I-{entity_type}'
                    break

        # Validate and fix BIO sequence
        tags = fix_bio_sequence(tags)
        bio_data.append((tokens, tags))

    return bio_data


def write_bio_format(
    data: List[Tuple[List[str], List[str]]],
    output_path: str,
    validate: bool = True
) -> Dict[str, int]:
    """
    Write data in BIO format to file.

    Args:
        data: List of (tokens, tags) tuples
        output_path: Path to output file
        validate: Whether to validate BIO sequences

    Returns:
        Statistics dictionary
    """
    stats = {
        'sentences': 0,
        'tokens': 0,
        'chemical_entities': 0,
        'disease_entities': 0,
        'fixed_sequences': 0
    }

    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        for tokens, tags in data:
            assert len(tokens) == len(tags), f"Length mismatch: {len(tokens)} tokens vs {len(tags)} tags"

            # Validate and fix if needed
            if validate:
                is_valid, _ = validate_bio_sequence(tags)
                if not is_valid:
                    tags = fix_bio_sequence(tags)
                    stats['fixed_sequences'] += 1

            for token, tag in zip(tokens, tags):
                f.write(f"{token}\t{tag}\n")

                # Count entities
                if tag.startswith('B-Chemical'):
                    stats['chemical_entities'] += 1
                elif tag.startswith('B-Disease'):
                    stats['disease_entities'] += 1

            f.write("\n")  # Blank line between sentences

            stats['sentences'] += 1
            stats['tokens'] += len(tokens)

    return stats


def split_data(
    data: List[Tuple[List[str], List[str]]],
    train_ratio: float = 0.7,
    dev_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_seed: int = 42,
    stratify: bool = True
) -> Tuple[List, List, List]:
    """
    Split data into train, dev, and test sets.

    Optionally stratifies to ensure entity distribution is similar across splits.

    Args:
        data: List of (tokens, tags) tuples
        train_ratio: Proportion for training set
        dev_ratio: Proportion for development set
        test_ratio: Proportion for test set
        random_seed: Random seed for reproducibility
        stratify: Whether to stratify by entity types

    Returns:
        (train_data, dev_data, test_data) tuple
    """
    assert abs(train_ratio + dev_ratio + test_ratio - 1.0) < 1e-6, \
        "Ratios must sum to 1.0"

    random.seed(random_seed)

    if stratify:
        # Group sentences by entity types present
        chemical_only = []
        disease_only = []
        both = []
        neither = []

        for tokens, tags in data:
            has_chemical = any(t.endswith('Chemical') for t in tags)
            has_disease = any(t.endswith('Disease') for t in tags)

            if has_chemical and has_disease:
                both.append((tokens, tags))
            elif has_chemical:
                chemical_only.append((tokens, tags))
            elif has_disease:
                disease_only.append((tokens, tags))
            else:
                neither.append((tokens, tags))

        # Split each group
        def split_group(group):
            random.shuffle(group)
            n = len(group)
            train_end = int(n * train_ratio)
            dev_end = train_end + int(n * dev_ratio)
            return group[:train_end], group[train_end:dev_end], group[dev_end:]

        train_data, dev_data, test_data = [], [], []

        for group in [both, chemical_only, disease_only, neither]:
            if group:
                tr, dv, te = split_group(group)
                train_data.extend(tr)
                dev_data.extend(dv)
                test_data.extend(te)

        # Shuffle final splits
        random.shuffle(train_data)
        random.shuffle(dev_data)
        random.shuffle(test_data)

    else:
        shuffled_data = data.copy()
        random.shuffle(shuffled_data)

        n = len(shuffled_data)
        train_end = int(n * train_ratio)
        dev_end = train_end + int(n * dev_ratio)

        train_data = shuffled_data[:train_end]
        dev_data = shuffled_data[train_end:dev_end]
        test_data = shuffled_data[dev_end:]

    return train_data, dev_data, test_data


def compute_class_weights(data: List[Tuple[List[str], List[str]]]) -> Dict[str, float]:
    """
    Compute class weights for handling imbalanced labels.

    Uses inverse frequency weighting.

    Args:
        data: List of (tokens, tags) tuples

    Returns:
        Dictionary mapping tags to weights
    """
    tag_counts = defaultdict(int)
    total = 0

    for tokens, tags in data:
        for tag in tags:
            tag_counts[tag] += 1
            total += 1

    # Compute inverse frequency weights
    num_classes = len(tag_counts)
    weights = {}

    for tag, count in tag_counts.items():
        # Weight = total / (num_classes * count)
        weights[tag] = total / (num_classes * count)

    # Normalize so minimum weight is 1.0
    min_weight = min(weights.values())
    weights = {tag: w / min_weight for tag, w in weights.items()}

    return weights


def preprocess_data(
    raw_dir: str = "data/raw",
    processed_dir: str = "data/processed",
    use_bc5cdr: bool = True,
    download: bool = False
) -> Dict[str, str]:
    """
    Main preprocessing function.

    Tries to use BC5CDR dataset if available, otherwise creates sample data.

    Args:
        raw_dir: Directory for raw data
        processed_dir: Directory for processed data
        use_bc5cdr: Whether to try using BC5CDR dataset
        download: Whether to download BC5CDR if not present

    Returns:
        Dictionary with paths to processed files
    """
    os.makedirs(processed_dir, exist_ok=True)

    print("=" * 70)
    print("BiLSTM-CRF NER Data Preprocessing")
    print("=" * 70)

    output_files = {}

    # Try BC5CDR first
    if use_bc5cdr:
        try:
            bc5cdr_files = process_bc5cdr_dataset(
                raw_dir,
                processed_dir,
                download=download
            )
            if bc5cdr_files:
                print("\nSuccessfully processed BC5CDR dataset!")
                return bc5cdr_files
        except Exception as e:
            print(f"\nCould not process BC5CDR dataset: {e}")

    # Fallback to sample data
    print("\nGenerating comprehensive sample biomedical NER data...")
    print("(For production use, download the BC5CDR dataset)")

    # Create sample data
    all_data = create_comprehensive_sample_data()

    print(f"\nGenerated {len(all_data)} unique sentences")

    # Verify no duplicates
    unique_sentences = set()
    for tokens, _ in all_data:
        sent_str = ' '.join(tokens)
        unique_sentences.add(sent_str)

    print(f"Unique sentences verified: {len(unique_sentences)}")

    # Split data with stratification
    train_data, dev_data, test_data = split_data(
        all_data,
        train_ratio=0.7,
        dev_ratio=0.15,
        test_ratio=0.15,
        random_seed=42,
        stratify=True
    )

    print(f"\nData splits:")
    print(f"  Train: {len(train_data)} sentences")
    print(f"  Dev:   {len(dev_data)} sentences")
    print(f"  Test:  {len(test_data)} sentences")

    # Write files
    splits = [
        ('train', train_data),
        ('dev', dev_data),
        ('test', test_data)
    ]

    for split_name, split_data_list in splits:
        output_path = os.path.join(processed_dir, f"{split_name}.txt")
        stats = write_bio_format(split_data_list, output_path)

        print(f"\n{split_name.upper()} set statistics:")
        print(f"  Sentences: {stats['sentences']}")
        print(f"  Tokens: {stats['tokens']}")
        print(f"  Chemical entities: {stats['chemical_entities']}")
        print(f"  Disease entities: {stats['disease_entities']}")

        output_files[split_name] = output_path

    # Compute and print class weights
    print("\nClass weights (for handling imbalance):")
    weights = compute_class_weights(train_data)
    for tag, weight in sorted(weights.items()):
        print(f"  {tag}: {weight:.2f}")

    # Print sample
    print("\n" + "-" * 70)
    print("Sample from train.txt:")
    print("-" * 70)
    with open(output_files['train'], 'r', encoding='utf-8') as f:
        lines = []
        for line in f:
            lines.append(line.rstrip())
            if line.strip() == '' and len(lines) > 10:
                break
    for line in lines[:15]:
        print(line)
    print("-" * 70)

    print("\nPreprocessing complete!")
    print("=" * 70)

    return output_files


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description='Preprocess NER data')
    parser.add_argument('--raw-dir', type=str, default='data/raw',
                        help='Directory for raw data')
    parser.add_argument('--processed-dir', type=str, default='data/processed',
                        help='Directory for processed data')
    parser.add_argument('--download', action='store_true',
                        help='Download BC5CDR dataset if not present')
    parser.add_argument('--sample-only', action='store_true',
                        help='Only generate sample data (skip BC5CDR)')

    args = parser.parse_args()

    preprocess_data(
        raw_dir=args.raw_dir,
        processed_dir=args.processed_dir,
        use_bc5cdr=not args.sample_only,
        download=args.download
    )


if __name__ == "__main__":
    main()
