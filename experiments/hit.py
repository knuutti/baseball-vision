import numpy as np

class HitAttributes:
    def __init__(self, speed: float = 0, angleH: float = 0, angleV: float = 0, spinV: float = 0, spinH: float = 0):
        """
        speed: float in range [0, inf) representing the speed of the ball in km/h
        angleH: float in range [-90, 90] representing the horizontal angle of the ball in degrees
        angleV: float in range [-90, 90] representing the vertical angle of the ball in degrees
        spinV: float in range (-inf, inf) representing the vertical spin of the ball in revolutions per second
        spinH: float in range (-inf, inf) representing the horizontal spin of the ball in revolutions per second
        """
        self.speed = speed
        self.angleH = angleH
        self.angleV = angleV
        self.spinV = spinV
        self.spinH = spinH

def calculate_air_resistance(speed: float, spinV: float, spinH: float) -> float:
    # Placeholder for air resistance calculation based on speed and spin
    # This is a simplified model; in reality, air resistance is more complex
    drag_coefficient = 0.47  # Approximate drag coefficient for a sphere
    air_density = 1.225  # kg/m^3 at sea level
    ball_radius = 0.22 / (2*3.14159)  # meters (standard tennis ball radius)
    ball_area = np.pi * ball_radius**2  # Cross-sectional area in m^2
    ball_mass = 0.057  # kg (standard tennis ball mass)

    speed_m_per_s = speed * 1000 / 3600  # Convert speed from km/h to m/s
    air_resistance_force = 0.5 * drag_coefficient * air_density * ball_area * speed_m_per_s**2

    return air_resistance_force

def calculate_end_position(hit_attributes: HitAttributes, initial_position: np.ndarray) -> np.ndarray:
    speed_m_per_s = hit_attributes.speed * 1000 / 3600  # Convert speed from km/h to m/s
    angleH_rad = np.radians(hit_attributes.angleH)
    angleV_rad = np.radians(hit_attributes.angleV)

    # Ignore air resistance and spin for simplicity, just calculate the end position based on speed and angles
    # Calculate how long ball travels until height is 0
    g = 9.81  # Acceleration due to gravity in m/s^2
    initial_height = initial_position[2] / 1000  # Convert mm to m

    time_of_flight_vacuum = (speed_m_per_s * np.sin(angleV_rad) + np.sqrt((speed_m_per_s * np.sin(angleV_rad))**2 + 2 * g * initial_height)) / g
    print(f"Time of flight: {time_of_flight_vacuum} seconds")
    # Calculate the horizontal distance traveled
    horizontal_distance = speed_m_per_s * np.cos(angleV_rad) * time_of_flight_vacuum * 1000  # Convert m to mm
    # Calculate air resistance effect (simplified)
    air_resistance = calculate_air_resistance(hit_attributes.speed, hit_attributes.spinV, hit_attributes.spinH)
    horizontal_distance -= air_resistance * time_of_flight_vacuum * 1000  # Adjust for air resistance
    # Calculate the end position
    end_x = initial_position[0] + horizontal_distance * np.cos(angleH_rad)
    end_y = initial_position[1] + horizontal_distance * np.sin(angleH_rad)
    end_z = 0
    end_position = np.array([end_x, end_y, end_z])

    return end_position

def hit_ball(hit_attributes: HitAttributes):
    initial_position = np.array([0, 0, 1200])  # Initial position of the ball in mm
    end_position = calculate_end_position(hit_attributes, initial_position)
    return end_position

if __name__ == "__main__":
    desired_hit = HitAttributes(speed=150, angleH=10, angleV=30, spinV=0, spinH=0)
    end_position = hit_ball(desired_hit)
    print(f"End position: {end_position/1000} m")