from matplotlib import patches
from matplotlib import transforms

class RotatingRectangle(patches.Rectangle):
    def __init__(self, xy, width: int, height: int, rel_point_of_rot=None, **kwargs):
        super().__init__(xy, width, height, **kwargs)
        self.rel_point_of_rot = (
            0, 0) if rel_point_of_rot is None else rel_point_of_rot
        self.xy_center = self.get_xy()

    def get_patch_transform(self):
        # Note: This cannot be called until after this has been added to
        # an Axes, otherwise unit conversion will fail. This makes it very
        # important to call the accessor method and not directly access the
        # transformation member variable.
        bbox = self.get_bbox()
        return (transforms.BboxTransformTo(bbox)
                + transforms.Affine2D().rotate_deg_around(
                    self.rel_point_of_rot[0], self.rel_point_of_rot[1], self.angle))