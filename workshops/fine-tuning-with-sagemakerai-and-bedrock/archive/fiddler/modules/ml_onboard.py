import fiddler as fdl
from typing import List


def create_country_segments(model_id: int, countries: List[str], prop_countries: List[str]):
    for country in prop_countries:
        try:
            fdl.Segment(
                name=f'Traveling to {country}',
                model_id=model_id,
                description=f'Search Queries for trips to {country}',
                definition=f'destination_country_id==\'{country}\'',
            ).create()
        except fdl.Conflict:
            print(f"Segment 'Traveling to {country}' already exists.")

    for country in countries:
        try:
            fdl.Segment(
                name=f'Visitor from {country}',
                model_id=model_id,
                description=f'Segment for visitors from {country}',
                definition=f'visitor_location_country_id==\'{country}\'',
            ).create()
        except fdl.Conflict:
            print(f"Segment 'Visitor from {country}' already exists.")

    predefined_segments = [
        (
            "No Click on Promo - USA",
            "Segment for visitors from USA with no click on promo",
            """user_interaction==0 and visitor_location_country_id=='USA'"""
        ),
        (
            "Clicked on Promo - USA",
            "Segment for visitors from USA who clicked on promo",
            """user_interaction==1 and visitor_location_country_id=='USA'"""
        ),
        (
            "Clicked Promo",
            "Segment for visitors who clicked on promo",
            """user_interaction==1"""
        ),
        (
            "No Click on Promo",
            "Segment for visitors with no click on promo",
            """user_interaction==0"""
        ),
    ]

    for name, description, definition in predefined_segments:
        try:
            fdl.Segment(
                name=name,
                model_id=model_id,
                description=description,
                definition=definition,
            ).create()
        except fdl.Conflict:
            print(f"Segment '{name}' already exists.")
