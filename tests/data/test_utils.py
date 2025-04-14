import pytest
import pandas as pd
import os
import json
from epibench.data.utils import fasta_to_json, bed_to_csv

# Helper function to create temporary files
@pytest.fixture
def temp_file(tmp_path):
    def _temp_file(content, suffix=".txt"):
        file = tmp_path / f"temp{suffix}"
        file.write_text(content)
        return str(file)
    return _temp_file

# --- Tests for fasta_to_json ---

def test_fasta_to_json_success(temp_file, tmp_path):
    fasta_content = ">seq1\nACGT\n>seq2\nNGCTA\n"
    fasta_path = temp_file(fasta_content, suffix=".fasta")
    json_path = str(tmp_path / "output.json")

    result = fasta_to_json(fasta_path, json_path)
    assert result is True
    assert os.path.exists(json_path)

    with open(json_path, 'r') as f:
        data = json.load(f)
    assert data == {"seq1": "ACGT", "seq2": "NGCTA"}

def test_fasta_to_json_file_not_found(tmp_path):
    fasta_path = str(tmp_path / "nonexistent.fasta")
    json_path = str(tmp_path / "output.json")
    result = fasta_to_json(fasta_path, json_path)
    assert result is False

def test_fasta_to_json_empty_fasta(temp_file, tmp_path):
    fasta_content = ""
    fasta_path = temp_file(fasta_content, suffix=".fasta")
    json_path = str(tmp_path / "output.json")

    result = fasta_to_json(fasta_path, json_path)
    assert result is True # Should succeed but produce empty JSON
    assert os.path.exists(json_path)
    with open(json_path, 'r') as f:
        data = json.load(f)
    assert data == {}

# --- Tests for bed_to_csv ---

def test_bed_to_csv_success_default_cols(temp_file, tmp_path):
    bed_content = "chr1\t100\t200\tfeature1\t60\t+\nchr1\t300\t400\tfeature2\t80\t-"
    bed_path = temp_file(bed_content, suffix=".bed")
    csv_path = str(tmp_path / "output.csv")

    result = bed_to_csv(bed_path, csv_path)
    assert result is True
    assert os.path.exists(csv_path)

    df = pd.read_csv(csv_path)
    assert list(df.columns) == ['chromosome', 'start', 'end', 'name', 'score', 'strand']
    assert len(df) == 2
    assert df.iloc[0]['chromosome'] == 'chr1'
    assert df.iloc[0]['start'] == 100
    assert df.iloc[0]['end'] == 200
    assert df.iloc[0]['name'] == 'feature1'
    assert df.iloc[0]['score'] == 60
    assert df.iloc[0]['strand'] == '+'


def test_bed_to_csv_success_custom_cols(temp_file, tmp_path):
    bed_content = "chrX\t10\t20\tgeneA\t1000\t+"
    bed_path = temp_file(bed_content, suffix=".bed")
    csv_path = str(tmp_path / "output.csv")
    custom_cols = ['chrom', 'chromStart', 'chromEnd', 'name', 'score', 'strand']

    result = bed_to_csv(bed_path, csv_path, column_names=custom_cols)
    assert result is True
    assert os.path.exists(csv_path)

    df = pd.read_csv(csv_path)
    assert list(df.columns) == custom_cols
    assert len(df) == 1
    assert df.iloc[0]['chrom'] == 'chrX'
    assert df.iloc[0]['score'] == 1000

def test_bed_to_csv_file_not_found(tmp_path):
    bed_path = str(tmp_path / "nonexistent.bed")
    csv_path = str(tmp_path / "output.csv")
    result = bed_to_csv(bed_path, csv_path)
    assert result is False

def test_bed_to_csv_missing_core_cols(temp_file, tmp_path):
    bed_content = "chr1\t100\nchr2\t200" # Missing end column
    bed_path = temp_file(bed_content, suffix=".bed")
    csv_path = str(tmp_path / "output.csv")
    result = bed_to_csv(bed_path, csv_path)
    assert result is False # Should fail validation

def test_bed_to_csv_non_numeric_coords(temp_file, tmp_path):
    bed_content = "chr1\tabc\t200\nchr2\t100\txyz"
    bed_path = temp_file(bed_content, suffix=".bed")
    csv_path = str(tmp_path / "output.csv")
    result = bed_to_csv(bed_path, csv_path)
    assert result is False # Should fail validation

def test_bed_to_csv_empty_file(temp_file, tmp_path):
    bed_content = ""
    bed_path = temp_file(bed_content, suffix=".bed")
    csv_path = str(tmp_path / "output.csv")
    result = bed_to_csv(bed_path, csv_path)
    assert result is True # Should succeed, create empty CSV
    assert os.path.exists(csv_path)
    df = pd.read_csv(csv_path)
    assert df.empty
