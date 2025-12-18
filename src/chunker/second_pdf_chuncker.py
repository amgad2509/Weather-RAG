import re
from typing import List
from langchain_core.documents import Document


class WeatherDatabaseBulletChunker:
    """
    For PDF_02 (Weather Activity Clothing Database) with format like:
      **1. Sunny Weather**
      Egypt:
        - Outdoor Activities: ...
        - Appropriate Clothing: ...
    """

    def __init__(self):
        self.weather_re = re.compile(
            r"(?m)^\s*\*{0,2}\s*(?P<num>\d+)\.\s+(?P<weather>.+?Weather)\s*\*{0,2}\s*$"
        )

        # Supports:
        # Egypt:
        # USA (California):
        # South Africa (Cape Town):
        self.country_re = re.compile(
            r"(?m)^\s*(?P<country>[A-Z][A-Za-z\s]+(?:\s*\([^)]+\))?)\s*:\s*$"
        )

        self.activities_re = re.compile(
            r"(?is)(?:^|\n)\s*(?:[-•]\s*)?Outdoor Activities\s*:\s*(?P<act>.*?)(?=\n\s*(?:[-•]\s*)?Appropriate Clothing\s*:|\Z)"
        )

        self.clothing_re = re.compile(
            r"(?is)(?:^|\n)\s*(?:[-•]\s*)?Appropriate Clothing\s*:\s*(?P<clo>.*?)(?=\Z)"
        )

    def split_text(self, text: str) -> List[Document]:
        docs: List[Document] = []
        if not text or not text.strip():
            return docs

        weather_sections = list(self.weather_re.finditer(text))
        if not weather_sections:
            return docs

        for i, w in enumerate(weather_sections):
            weather_type = (w.group("weather") or "").strip()

            w_start = w.end()
            w_end = weather_sections[i + 1].start() if i + 1 < len(weather_sections) else len(text)
            weather_block = text[w_start:w_end].strip()

            countries = list(self.country_re.finditer(weather_block))
            if not countries:
                continue

            for j, c in enumerate(countries):
                country = (c.group("country") or "").strip()

                c_start = c.end()
                c_end = countries[j + 1].start() if j + 1 < len(countries) else len(weather_block)
                country_block = weather_block[c_start:c_end].strip()
                if not country_block:
                    continue

                am = self.activities_re.search(country_block)
                cm = self.clothing_re.search(country_block)

                activities = (am.group("act").strip() if am else "").strip()
                clothing = (cm.group("clo").strip() if cm else "").strip()

                # If no clear subsection matches, keep whole block as one chunk
                if not activities and not clothing:
                    docs.append(
                        Document(
                            page_content=country_block,
                            metadata={
                                "country": country,
                                "weather": weather_type,
                                "section": "full",
                                "source": "weather_database",
                            },
                        )
                    )
                    continue

                if activities:
                    docs.append(
                        Document(
                            page_content="Outdoor Activities:\n" + activities,
                            metadata={
                                "country": country,
                                "weather": weather_type,
                                "section": "Outdoor Activities",
                                "source": "weather_database",
                            },
                        )
                    )

                if clothing:
                    docs.append(
                        Document(
                            page_content="Appropriate Clothing:\n" + clothing,
                            metadata={
                                "country": country,
                                "weather": weather_type,
                                "section": "Appropriate Clothing",
                                "source": "weather_database",
                            },
                        )
                    )

        return docs