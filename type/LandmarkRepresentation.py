from enum import Enum, auto

class LandmarkRepresentation:
    """
    Class holding useful feature representation types.
    """
    
    class Representation(Enum):
        """
        What feature representation our state can use.
        """
        GLOBAL_3D = auto()
        GLOBAL_FULL_INVERSE_DEPTH = auto()
        ANCHORED_3D = auto()
        ANCHORED_FULL_INVERSE_DEPTH = auto()
        ANCHORED_MSCKF_INVERSE_DEPTH = auto()
        ANCHORED_INVERSE_DEPTH_SINGLE = auto()
        UNKNOWN = auto()

    @staticmethod
    def as_string(feat_representation: 'LandmarkRepresentation.Representation') -> str:
        """
        Returns a string representation of this enum value.
        :param feat_representation: Representation we want to check
        :return: String version of the passed enum
        """
        if feat_representation == LandmarkRepresentation.Representation.GLOBAL_3D:
            return "GLOBAL_3D"
        if feat_representation == LandmarkRepresentation.Representation.GLOBAL_FULL_INVERSE_DEPTH:
            return "GLOBAL_FULL_INVERSE_DEPTH"
        if feat_representation == LandmarkRepresentation.Representation.ANCHORED_3D:
            return "ANCHORED_3D"
        if feat_representation == LandmarkRepresentation.Representation.ANCHORED_FULL_INVERSE_DEPTH:
            return "ANCHORED_FULL_INVERSE_DEPTH"
        if feat_representation == LandmarkRepresentation.Representation.ANCHORED_MSCKF_INVERSE_DEPTH:
            return "ANCHORED_MSCKF_INVERSE_DEPTH"
        if feat_representation == LandmarkRepresentation.Representation.ANCHORED_INVERSE_DEPTH_SINGLE:
            return "ANCHORED_INVERSE_DEPTH_SINGLE"
        return "UNKNOWN"

    @staticmethod
    def from_string(feat_representation: str) -> 'LandmarkRepresentation.Representation':
        """
        Returns the enum value corresponding to the passed string.
        :param feat_representation: String we want to find the enum of
        :return: Representation enum, will be UNKNOWN if we couldn't parse it
        """
        if feat_representation == "GLOBAL_3D":
            return LandmarkRepresentation.Representation.GLOBAL_3D
        if feat_representation == "GLOBAL_FULL_INVERSE_DEPTH":
            return LandmarkRepresentation.Representation.GLOBAL_FULL_INVERSE_DEPTH
        if feat_representation == "ANCHORED_3D":
            return LandmarkRepresentation.Representation.ANCHORED_3D
        if feat_representation == "ANCHORED_FULL_INVERSE_DEPTH":
            return LandmarkRepresentation.Representation.ANCHORED_FULL_INVERSE_DEPTH
        if feat_representation == "ANCHORED_MSCKF_INVERSE_DEPTH":
            return LandmarkRepresentation.Representation.ANCHORED_MSCKF_INVERSE_DEPTH
        if feat_representation == "ANCHORED_INVERSE_DEPTH_SINGLE":
            return LandmarkRepresentation.Representation.ANCHORED_INVERSE_DEPTH_SINGLE
        return LandmarkRepresentation.Representation.UNKNOWN

    @staticmethod
    def is_relative_representation(feat_representation: 'LandmarkRepresentation.Representation') -> bool:
        """
        Helper function that checks if the passed feature representation is a relative or global.
        :param feat_representation: Representation we want to check
        :return: True if it is a relative representation
        """
        return (feat_representation == LandmarkRepresentation.Representation.ANCHORED_3D or 
                feat_representation == LandmarkRepresentation.Representation.ANCHORED_FULL_INVERSE_DEPTH or
                feat_representation == LandmarkRepresentation.Representation.ANCHORED_MSCKF_INVERSE_DEPTH or
                feat_representation == LandmarkRepresentation.Representation.ANCHORED_INVERSE_DEPTH_SINGLE)

    def __init__(self):
        """
        Private constructor to prevent instantiation (static class).
        """
        raise NotImplementedError("This is a static utility class and cannot be instantiated.")