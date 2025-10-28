from pydantic import BaseModel

FEATURES = [
    'Elevation', 'Aspect', 'Slope', 'Horizontal_Distance_To_Hydrology',
    'Vertical_Distance_To_Hydrology', 'Horizontal_Distance_To_Roadways',
    'Hillshade_9am', 'Hillshade_Noon', 'Hillshade_3pm',
    'Horizontal_Distance_To_Fire_Points', 'Wilderness_Area_01',
    'Wilderness_Area_02', 'Wilderness_Area_03', 'Wilderness_Area_04',
    'Soil_Type_01', 'Soil_Type_02', 'Soil_Type_03', 'Soil_Type_04',
    'Soil_Type_05', 'Soil_Type_06', 'Soil_Type_07', 'Soil_Type_08',
    'Soil_Type_09', 'Soil_Type_10', 'Soil_Type_11', 'Soil_Type_12',
    'Soil_Type_13', 'Soil_Type_14', 'Soil_Type_15', 'Soil_Type_16',
    'Soil_Type_17', 'Soil_Type_18', 'Soil_Type_19', 'Soil_Type_20',
    'Soil_Type_21', 'Soil_Type_22', 'Soil_Type_23', 'Soil_Type_24',
    'Soil_Type_25', 'Soil_Type_26', 'Soil_Type_27', 'Soil_Type_28',
    'Soil_Type_29', 'Soil_Type_30', 'Soil_Type_31', 'Soil_Type_32',
    'Soil_Type_33', 'Soil_Type_34', 'Soil_Type_35', 'Soil_Type_36',
    'Soil_Type_37', 'Soil_Type_38', 'Soil_Type_39', 'Soil_Type_40'
]

class CoverTypePayload(BaseModel):
    """
    Schema for the input features payload for model inference.
    All fields are expected to be numeric (float or int).
    """
    # Dynamically define fields based on the FEATURES list
    # Use a dictionary comprehension to create fields with float type
    __annotations__ = {f: int for f in FEATURES}

class CoverTypeResponse(BaseModel):
    """
    Schema for the model prediction response.
    """
    # The output field as requested
    cover_type: int
    percentages: list[float]