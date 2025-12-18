import re
from typing import List
from langchain_core.documents import Document


class StructuredWeatherClothingChunker:
    """
    For PDF_01 (Comprehensive Global Guide) with format like:
      ## **1. Clear/Sunny Weather**
      ### **Egypt**
      ...
      **Temperature Range:**
      - High: ...
      - Low: ...
    """

    def __init__(self):
        self.weather_re = re.compile(
            r"(?m)^\s*##\s+\*\*(?P<title>\d+\.\s+.+?)\*\*\s*$"
        )
        self.country_re = re.compile(
            r"(?m)^\s*###\s+\*\*(?P<country>.+?)\*\*\s*$"
        )
        self.temp_re = re.compile(
            r"(?is)\*\*Temperature Range:\*\*\s*(?P<temp>.*?)(?=\n\s*###\s+\*\*|\n\s*##\s+\*\*|\Z)"
        )

    def split_text(self, text: str) -> List[Document]:
        documents: List[Document] = []
        if not text or not text.strip():
            return documents

        weather_sections = list(self.weather_re.finditer(text))
        if not weather_sections:
            return documents

        for i, w in enumerate(weather_sections):
            raw_weather = (w.group("title") or "").strip()
            weather_type = re.sub(r"^\d+\.\s*", "", raw_weather).strip()

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

                # extract temperature range
                temp_text = ""
                clean_content = country_block

                tm = self.temp_re.search(country_block)
                if tm:
                    temp_text = (tm.group("temp") or "").strip()
                    clean_content = country_block[:tm.start()].strip()

                # Skip empty
                if not clean_content:
                    continue

                documents.append(
                    Document(
                        page_content=clean_content,
                        metadata={
                            "country": country,
                            "weather": weather_type,
                            "temperature_range": temp_text,
                            "source": "comprehensive_global_guide",
                        },
                    )
                )

        return documents
