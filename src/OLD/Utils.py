import numpy as np

class CFrameND:
    """
    An N-dimensional Coordinate Frame, inspired by Roblox's CFrame.
    It encapsulates an N-D position and an N-D orientation (basis matrix).
    """
    def __init__(self, position: np.ndarray, basis: np.ndarray):
        self.dims = position.shape[0]
        if basis.shape != (self.dims, self.dims):
            raise ValueError("Shape of basis matrix must match dimensions of position vector.")
        
        # Ensure data types are consistent for performance
        self.position = position.astype(np.float64)
        self.basis = basis.astype(np.float64)

    @staticmethod
    def identity(dims):
        """Creates a CFrame at the origin with no rotation."""
        pos = np.zeros(dims)
        basis = np.identity(dims)
        return CFrameND(pos, basis)

    @staticmethod
    def from_position(pos_vector):
        """Creates a CFrame at a given position with no rotation."""
        dims = len(pos_vector)
        basis = np.identity(dims)
        return CFrameND(np.array(pos_vector), basis)

    @staticmethod
    def from_rotation_plane(b_idx1, b_idx2, angle, dims):
        """Creates a pure rotational CFrame."""
        pos = np.zeros(dims)
        basis = np.identity(dims)
        c, s = np.cos(angle), np.sin(angle)
        
        # Apply the 2D rotation to the basis
        basis[b_idx1, b_idx1] = c
        basis[b_idx1, b_idx2] = -s
        basis[b_idx2, b_idx1] = s
        basis[b_idx2, b_idx2] = c
        return CFrameND(pos, basis)

    def __mul__(self, other):
        """
        The core transformation operator. Overloads the '*' symbol.
        - CFrame * CFrame:  Applies a relative transformation.
        - CFrame * Vector:  Transforms a local point to a world point.
        """
        # Case 1: CFrame * CFrame (Combining transformations)
        if isinstance(other, CFrameND):
            # The new orientation is the product of the basis matrices.
            new_basis = self.basis @ other.basis
            # The new position is self.pos + other.pos transformed by self's orientation.
            new_pos = self.position + self.basis @ other.position
            return CFrameND(new_pos, new_basis)

        # Case 2: CFrame * Vector (Transforming a point from local to world space)
        if isinstance(other, np.ndarray):
            # Start at the CFrame's position and add the rotated local vector.
            return self.position + self.basis @ other
        
        return NotImplemented

    def inverse(self):
        """Returns the inverse transformation."""
        # The inverse of an orthogonal rotation matrix is its transpose.
        inv_basis = self.basis.T
        # The inverse position is the negative of the original position,
        # transformed by the inverse rotation.
        inv_pos = -inv_basis @ self.position
        return CFrameND(inv_pos, inv_basis)

    def to_object_space(self, world_cframe):
        """Transforms a world CFrame into this CFrame's local space."""
        # This is the core formula: A:ToObjectSpace(B) == A:Inverse() * B
        return self.inverse() * world_cframe
    
    def __repr__(self):
        return f"CFrameND(dims={self.dims}, pos={self.position})"


def GetPlane(domains: tuple, screensize: tuple):
    Plane = np.array(np.meshgrid(np.linspace(-1,1,screensize[0]),np.linspace(-1,1,screensize[1])))
    return Plane

"""def



print(GetPlane(16,9)) """

"HELLO"