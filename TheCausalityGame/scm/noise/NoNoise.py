# # Abstract
# from .abstract import BaseNoiseDistribution

# # Math
# import numpy as np


# class NoNoiseDistribution(BaseNoiseDistribution):
#     def __init__(self):
#         """
#         Initializes the Gaussian noise distribution with a given mean and standard deviation.

#         Args:
#             mean (float): The mean of the Gaussian distribution.
#             std (float): The standard deviation of the Gaussian distribution.
#         """

#     def generate(self, size, random_state: int = 911) -> float:
#         """
#         Generates a noise value using the Gaussian distribution.

#         Args:
#             random_state (int, optional): Seed for random number generation. Defaults to 911.

#         Returns:
#             float: A generated noise value.
#         """
#         return np.zeros(size)

#     def to_dict(self) -> dict:
#         """
#         Serializes the Gaussian noise object into a dictionary format.

#         Returns:
#             dict: The dictionary representation of the Gaussian noise object.
#         """
#         return {
#             "class": NoNoiseDistribution.__name__,
#         }

#     @classmethod
#     def from_dict(cls, data: dict) -> "NoNoiseDistribution":
#         """
#         Deserializes the No noise object from a dictionary representation.

#         Args:
#             data (dict): The dictionary containing No noise data.

#         Returns:
#             NoNoiseDistribution: An instance of No noise distribution.
#         """
#         return cls()
