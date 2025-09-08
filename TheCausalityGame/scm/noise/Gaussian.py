# # Abstract
# from .abstract import BaseNoiseDistribution

# # Distributions
# from scipy.stats import norm


# class GaussianNoiseDistribution(BaseNoiseDistribution):
#     def __init__(self, mean: float = 0.0, std: float = 1.0):
#         """
#         Initializes the Gaussian noise distribution with a given mean and standard deviation.

#         Args:
#             mean (float): The mean of the Gaussian distribution.
#             std (float): The standard deviation of the Gaussian distribution.
#         """
#         self.mean = mean
#         self.std = std

#     def generate(self, size, random_state: int = 911) -> float:
#         """
#         Generates a noise value using the Gaussian distribution.

#         Args:
#             random_state (int, optional): Seed for random number generation. Defaults to 911.

#         Returns:
#             float: A generated noise value.
#         """
#         return norm.rvs(
#             loc=self.mean, scale=self.std, size=size, random_state=random_state
#         )

#     def to_dict(self) -> dict:
#         """
#         Serializes the Gaussian noise object into a dictionary format.

#         Returns:
#             dict: The dictionary representation of the Gaussian noise object.
#         """
#         return {
#             "class": GaussianNoiseDistribution.__name__,
#             "mean": self.mean,
#             "std": self.std,
#         }

#     @classmethod
#     def from_dict(cls, data: dict) -> "GaussianNoiseDistribution":
#         """
#         Deserializes the Gaussian noise object from a dictionary representation.

#         Args:
#             data (dict): The dictionary containing Gaussian noise data.

#         Returns:
#             GaussianNoiseDistribution: An instance of Gaussian noise distribution.
#         """
#         return cls(mean=data["mean"], std=data["std"])
