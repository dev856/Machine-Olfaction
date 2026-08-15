"""Semantic odor profiles and chemical volatile metadata for SmellNet.

Connects machine olfaction sensor classifications to real physical chemistry,
botanical classifications, primary volatile organic compounds (VOCs),
and sensory flavor/aroma notes.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Default path to SmellNet text metadata
DEFAULT_TEXT_PATH = Path(__file__).resolve().parents[2] / "data" / "raw" / "SmellNet" / "text_data" / "text_description.json"

# Curated categories for the 50 SmellNet classes
ODOR_CATEGORIES: dict[str, str] = {
    # Fruits & Citrus
    "apple": "Fruits & Citrus",
    "banana": "Fruits & Citrus",
    "kiwi": "Fruits & Citrus",
    "lemon": "Fruits & Citrus",
    "mandarin_orange": "Fruits & Citrus",
    "mango": "Fruits & Citrus",
    "peach": "Fruits & Citrus",
    "pear": "Fruits & Citrus",
    "pineapple": "Fruits & Citrus",
    "strawberry": "Fruits & Citrus",
    "tomato": "Fruits & Citrus",
    "avocado": "Fruits & Citrus",
    # Spices
    "allspice": "Warm Spices",
    "cinnamon": "Warm Spices",
    "cloves": "Warm Spices",
    "coriander": "Warm Spices",
    "cumin": "Warm Spices",
    "ginger": "Warm Spices",
    "mustard": "Warm Spices",
    "nutmeg": "Warm Spices",
    "saffron": "Warm Spices",
    "star_anise": "Warm Spices",
    # Herbs & Botanicals
    "angelica": "Herbs & Botanicals",
    "chamomile": "Herbs & Botanicals",
    "chervil": "Herbs & Botanicals",
    "chives": "Herbs & Botanicals",
    "dill": "Herbs & Botanicals",
    "mint": "Herbs & Botanicals",
    "mugwort": "Herbs & Botanicals",
    "oregano": "Herbs & Botanicals",
    # Nuts & Seeds
    "almond": "Nuts & Seeds",
    "brazil_nut": "Nuts & Seeds",
    "cashew": "Nuts & Seeds",
    "chestnuts": "Nuts & Seeds",
    "hazelnut": "Nuts & Seeds",
    "peanuts": "Nuts & Seeds",
    "pecans": "Nuts & Seeds",
    "pili_nut": "Nuts & Seeds",
    "pistachios": "Nuts & Seeds",
    "walnuts": "Nuts & Seeds",
    # Cruciferous & Allium Vegetables (Sulfur-rich)
    "asparagus": "Cruciferous & Sulfuric",
    "broccoli": "Cruciferous & Sulfuric",
    "brussel_sprouts": "Cruciferous & Sulfuric",
    "cabbage": "Cruciferous & Sulfuric",
    "cauliflower": "Cruciferous & Sulfuric",
    "garlic": "Cruciferous & Sulfuric",
    "radish": "Cruciferous & Sulfuric",
    "turnip": "Cruciferous & Sulfuric",
    # Root & Tuber Vegetables
    "potato": "Roots & Tubers",
    "sweet_potato": "Roots & Tubers",
}

# Curated primary key chemical volatiles per class
KEY_VOLATILES: dict[str, list[str]] = {
    "allspice": ["Eugenol", "Caryophyllene", "1,8-Cineole", "Phellandrene"],
    "almond": ["Benzaldehyde", "Hexanal", "Oleic Acid derivatives"],
    "angelica": ["α-Pinene", "β-Phellandrene", "Limonene", "Osthole"],
    "apple": ["Hexyl Acetate", "2-Methylbutyl Acetate", "Ethyl Butyrate", "Hexanal"],
    "asparagus": ["Methanethiol", "Dimethyl Sulfide", "Asparagusic Acid"],
    "avocado": ["Hexanal", "(E)-2-Hexenal", "Nonanal", "Octanal"],
    "banana": ["Isoamyl Acetate", "Hexyl Acetate", "1-Butanol", "Eugenol"],
    "brazil_nut": ["Methylpyrazine", "2-Pentylfuran", "Dimethyl Disulfide"],
    "broccoli": ["Dimethyl Trisulfide", "Methanethiol", "Allyl Isothiocyanate", "Sulforaphane"],
    "brussel_sprouts": ["Glucosinolates", "Dimethyl Sulfide", "Allyl Isothiocyanate"],
    "cabbage": ["Methanethiol", "Dimethyl Disulfide", "Isothiocyanates"],
    "cashew": ["Hexanal", "Nonanal", "Alkylpyrazines"],
    "cauliflower": ["Dimethyl Trisulfide", "Hydrogen Sulfide", "Isothiocyanates"],
    "chamomile": ["α-Bisabolol", "Chamazulene", "Apigenin derivatives"],
    "chervil": ["Estragole (Methyl Chavicol)", "1-Allyl-2,4-dimethoxybenzene"],
    "chestnuts": ["Furfural", "Maltol", "Alkylpyrazines"],
    "chives": ["Allyl Sulfides", "Dipropyl Disulfide", "Thiosulfinates"],
    "cinnamon": ["Cinnamaldehyde", "Eugenol", "Coumarin", "Linalool"],
    "cloves": ["Eugenol", "Acetyl Eugenol", "β-Caryophyllene", "Humulene"],
    "coriander": ["Linalool", "γ-Terpinene", "Camphor", "α-Pinene"],
    "cumin": ["Cuminaldehyde", "γ-Terpinene", "β-Pinene", "p-Cymene"],
    "dill": ["Carvone", "Limonene", "α-Phellandrene"],
    "garlic": ["Allicin", "Diallyl Disulfide", "Diallyl Trisulfide", "Ajoene"],
    "ginger": ["Zingiberene", "β-Sesquiphellandrene", "Citral", "Gingerols"],
    "hazelnut": ["Filbertone", "2-Acetyl-1-pyrroline", "Hexanal"],
    "kiwi": ["Ethyl Butanoate", "Hexanal", "(E)-2-Hexenal", "Linalool"],
    "lemon": ["Limonene", "Citral", "β-Pinene", "γ-Terpinene"],
    "mandarin_orange": ["Limonene", "γ-Terpinene", "Linalool", "Myrcene"],
    "mango": ["δ-3-Carene", "Myrcene", "Limonene", "Linalool", "Lactones"],
    "mint": ["Menthol", "Menthone", "Limonene", "1,8-Cineole"],
    "mugwort": ["Camphor", "1,8-Cineole", "Thujone", "Borneol"],
    "mustard": ["Allyl Isothiocyanate", "Butenyl Isothiocyanate", "Sinigrin"],
    "nutmeg": ["Sabinene", "Myristicin", "Eugenol", "Safrole"],
    "oregano": ["Carvacrol", "Thymol", "p-Cymene", "γ-Terpinene"],
    "peach": ["γ-Decalactone", "γ-Undecalactone", "Hexyl Acetate", "Linalool"],
    "peanuts": ["2,5-Dimethylpyrazine", "2-Ethylpyrazine", "Hexanal"],
    "pear": ["Ethyl Hexanoate", "Hexyl Acetate", "Ethyl Decadienoate"],
    "pecans": ["Maltol", "Furfural", "Alkylpyrazines", "Lactones"],
    "pili_nut": ["Oleic Acid derivatives", "Alkylpyrazines", "Lactones"],
    "pineapple": ["Ethyl Butyrate", "Methyl Hexanoate", "Furaneol", "Acetaldehyde"],
    "pistachios": ["2-Methylpyrazine", "Hexanal", "Nonanal", "α-Pinene"],
    "potato": ["Methional", "2-Isopropyl-3-methoxypyrazine", "Hexanal"],
    "radish": ["4-Methylthio-3-butenyl Isothiocyanate", "Methanethiol", "Dimethyl Disulfide"],
    "saffron": ["Safranal", "Picrocrocin", "Crocin"],
    "star_anise": ["trans-Anethole", "Estragole", "Limonene", "Linalool"],
    "strawberry": ["Ethyl Butanoate", "Methyl Butanoate", "Furaneol", "Linalool"],
    "sweet_potato": ["Hexanal", "Nonanal", "Furan derivatives", "Lactones"],
    "tomato": ["cis-3-Hexenal", "cis-3-Hexenol", "β-Ionone", "6-Methyl-5-hepten-2-one"],
    "turnip": ["Dimethyl Sulfide", "Isothiocyanates", "Methanethiol"],
    "walnuts": ["Juglone", "1-Octen-3-ol", "Hexanal", "Pentanal"],
}

# Sensory aroma descriptors per class
SENSORY_NOTES: dict[str, list[str]] = {
    "allspice": ["warm-spicy", "clove-nutmeg nuance", "balsamic", "pungent"],
    "almond": ["sweet-nutty", "marzipan", "buttery", "clean"],
    "angelica": ["musky-green", "earthy", "resinous", "herbaceous"],
    "apple": ["crisp-fruity", "fresh-estery", "sweet-tart", "juicy"],
    "asparagus": ["sulfuric", "grassy-vegetal", "earthy", "sharp"],
    "avocado": ["creamy", "grassy-buttery", "subdued", "mild"],
    "banana": ["sweet-estery", "candy-like", "tropical", "rich"],
    "brazil_nut": ["rich-nutty", "oily", "warm-roasted", "earthy"],
    "broccoli": ["pungent-sulfuric", "cruciferous-green", "earthy"],
    "brussel_sprouts": ["sulfur-rich", "pungent", "bitter-green", "savory"],
    "cabbage": ["mild-sulfuric", "green-vegetal", "sweet-savory"],
    "cashew": ["creamy-buttery", "roasted", "delicate", "sweet"],
    "cauliflower": ["pungent-sulfuric", "nutty-cabbage", "mild-sweet"],
    "chamomile": ["sweet-apple", "floral-honey", "calming", "herbaceous"],
    "chervil": ["mild-anise", "sweet-parsley", "licorice nuance", "fresh"],
    "chestnuts": ["roasted-caramel", "starchy-sweet", "warm-nutty"],
    "chives": ["fresh-pungent", "onion-sulfuric", "sharp-green"],
    "cinnamon": ["sweet-warm", "woody-spicy", "aldehyde-rich", "intense"],
    "cloves": ["intensely-spicy", "medicinal-eugenol", "warm", "woody"],
    "coriander": ["warm-citrusy", "floral-linalool", "nutty-spicy"],
    "cumin": ["warm-earthy", "smoky-spicy", "aldehydic", "distinctive"],
    "dill": ["tangy-grassy", "fresh-herbal", "citrusy-anise"],
    "garlic": ["intense-sulfuric", "sharp-acrid", "pungent", "allicin-rich"],
    "ginger": ["spicy-peppery", "warm-citrusy", "woody-pungent"],
    "hazelnut": ["roasted-buttery", "warm-nutty", "sweet-caramelized"],
    "kiwi": ["sweet-tart", "green-tropical", "ester-fresh"],
    "lemon": ["sharp-zesty", "citrus-limonene", "refreshing", "bright"],
    "mandarin_orange": ["sweet-citrus", "floral-candy", "juicy-bright"],
    "mango": ["tropical-fruity", "resinous-green", "creamy-sweet"],
    "mint": ["cooling-menthol", "crisp-refreshing", "camphoraceous"],
    "mugwort": ["earthy-bitter", "medicinal", "camphor-sage", "herbaceous"],
    "mustard": ["sharp-pungent", "nose-tingling", "spicy-sulfuric"],
    "nutmeg": ["sweet-spicy", "warm-woody", "medicinal-aromatic"],
    "oregano": ["warm-peppery", "herbaceous-phenolic", "antimicrobial-sharp"],
    "peach": ["creamy-fruity", "lactonic-sweet", "floral-fresh"],
    "peanuts": ["roasted-nutty", "pyrazine-savory", "earthy"],
    "pear": ["delicate-sweet", "soft-floral", "crisp-estery"],
    "pecans": ["buttery-caramel", "warm-roasted", "sweet-nutty"],
    "pili_nut": ["creamy-buttery", "mild-nutty", "smooth"],
    "pineapple": ["vibrant-tangy", "tropical-estery", "sweet-acidic"],
    "pistachios": ["sweet-nutty", "green-roasted", "creamy"],
    "potato": ["mild-earthy", "starchy-savory", "grassy"],
    "radish": ["sharp-peppery", "pungent-sulfuric", "crisp"],
    "saffron": ["hay-like", "honeyed-metallic", "warm-floral"],
    "star_anise": ["sweet-licorice", "anethole-rich", "warm-spicy"],
    "strawberry": ["sweet-candy", "rich-estery", "floral-fresh"],
    "sweet_potato": ["caramelized-sweet", "roasted-earthy", "warm"],
    "tomato": ["fresh-green", "savory-sweet", "leafy-aldehyde"],
    "turnip": ["sharp-cabbage", "sulfuric-pungent", "earthy"],
    "walnuts": ["earthy-woody", "bitter-nutty", "astringent"],
}


@dataclass(frozen=True)
class SemanticOdorProfile:
    name: str
    category: str
    summary: str
    volatiles: list[str]
    sensory_notes: list[str]
    full_description: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "category": self.category,
            "summary": self.summary,
            "volatiles": self.volatiles,
            "sensory_notes": self.sensory_notes,
            "full_description": self.full_description,
        }


class SemanticKnowledgeBase:
    """In-memory registry of semantic odor profiles and chemical VOC descriptors."""

    def __init__(self, json_path: Path | str | None = None) -> None:
        path = Path(json_path) if json_path else DEFAULT_TEXT_PATH
        self._profiles: dict[str, SemanticOdorProfile] = {}
        self._raw_descriptions: dict[str, str] = {}
        if path.exists():
            try:
                with open(path, "r", encoding="utf-8") as f:
                    self._raw_descriptions = json.load(f)
            except Exception:
                self._raw_descriptions = {}

        self._build_profiles()

    def _build_profiles(self) -> None:
        all_classes = set(self._raw_descriptions.keys()) | set(ODOR_CATEGORIES.keys())
        for class_name in sorted(all_classes):
            desc = self._raw_descriptions.get(class_name, f"Odor class: {class_name.replace('_', ' ').title()}.")
            # Extract first sentence as summary
            first_sentence = desc.split(".")[0].strip() + "." if "." in desc else desc
            cat = ODOR_CATEGORIES.get(class_name, "Uncategorized")
            volatiles = KEY_VOLATILES.get(class_name, ["Volatile Organic Compounds (VOCs)"])
            sensory = SENSORY_NOTES.get(class_name, ["Aromatic"])

            self._profiles[class_name] = SemanticOdorProfile(
                name=class_name,
                category=cat,
                summary=first_sentence,
                volatiles=volatiles,
                sensory_notes=sensory,
                full_description=desc,
            )

    def get_profile(self, smell_class: str) -> SemanticOdorProfile:
        norm = smell_class.strip().lower().replace(" ", "_")
        if norm in self._profiles:
            return self._profiles[norm]

        # Return a sensible fallback profile
        return SemanticOdorProfile(
            name=norm,
            category=ODOR_CATEGORIES.get(norm, "General Odorant"),
            summary=f"Gas sensor signature for {norm.replace('_', ' ').title()}.",
            volatiles=KEY_VOLATILES.get(norm, ["Unspecified Volatiles"]),
            sensory_notes=SENSORY_NOTES.get(norm, ["Detectable"]),
            full_description=f"Sensory recording profile for smell class {norm.replace('_', ' ').title()}.",
        )

    def list_all_classes(self) -> list[str]:
        return sorted(list(self._profiles.keys()))

    def list_categories(self) -> list[str]:
        return sorted(list({p.category for p in self._profiles.values()}))

    def get_by_category(self, category: str) -> list[SemanticOdorProfile]:
        return [p for p in self._profiles.values() if p.category.lower() == category.lower()]


# Global singleton instance
_GLOBAL_KB: SemanticKnowledgeBase | None = None


def get_knowledge_base(json_path: Path | str | None = None) -> SemanticKnowledgeBase:
    global _GLOBAL_KB
    if _GLOBAL_KB is None or json_path is not None:
        _GLOBAL_KB = SemanticKnowledgeBase(json_path)
    return _GLOBAL_KB


def get_semantic_profile(smell_class: str) -> SemanticOdorProfile:
    return get_knowledge_base().get_profile(smell_class)
