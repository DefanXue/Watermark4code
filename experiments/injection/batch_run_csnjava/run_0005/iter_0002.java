public static double distance(LineSegment3D_F64 seg,
								   Point3D_F64 pt) {

		double dx = pt.x - seg.a.x;
		double dy = pt.y - seg.a.y;
		double dz = pt.z - seg.a.z;

		double segDx = seg.b.x - seg.a.x;
		double segDy = seg.b.y - seg.a.y;
		double segDz = seg.b.z - seg.a.z;

		double segmentLengthSq = segDx * segDx + segDy * segDy + segDz * segDz;

		if (segmentLengthSq == 0.0) {
			return pt.distance(seg.a);
		}

		double dotProduct = dx * segDx + dy * segDy + dz * segDz;
		double t = dotProduct / segmentLengthSq;

		double closestX, closestY, closestZ;

		if (t < 0.0) {
			closestX = seg.a.x;
			closestY = seg.a.y;
			closestZ = seg.a.z;
		} else if (t > 1.0) {
			closestX = seg.b.x;
			closestY = seg.b.y;
			closestZ = seg.b.z;
		} else {
			closestX = seg.a.x + t * segDx;
			closestY = seg.a.y + t * segDy;
			closestZ = seg.a.z + t * segDz;
		}

		double distSq = (pt.x - closestX) * (pt.x - closestX) +
						(pt.y - closestY) * (pt.y - closestY) +
						(pt.z - closestZ) * (pt.z - closestZ);

		return Math.sqrt(distSq);
	}