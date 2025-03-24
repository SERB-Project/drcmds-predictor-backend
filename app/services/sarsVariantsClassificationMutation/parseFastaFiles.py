def parse_fasta(fasta_content: str) -> str:
    """Extract sequence from FASTA format, ignoring header lines."""
    lines = fasta_content.strip().split('\n')
    # Skip header lines (those starting with '>')
    sequence_lines = [line for line in lines if not line.startswith('>')]
    return ''.join(sequence_lines)