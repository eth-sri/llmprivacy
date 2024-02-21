from typing import List

from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine


def anonymize_presidio(
    to_anon: str,
    entities: List[str] = [
        "PERSON",
        "EMAIL_ADDRESS",
        "PHONE_NUMBER",
        "LOCATION",
        "NRP",
    ],
) -> str:
    analyzer = AnalyzerEngine()
    anonymizer = AnonymizerEngine()
    res = analyzer.analyze(text=to_anon, entities=entities, language="en")
    anon_text = anonymizer.anonymize(
        text=to_anon,
        analyzer_results=res,
    )

    return anon_text
