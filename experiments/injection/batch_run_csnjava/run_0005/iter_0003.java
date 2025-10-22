public static double distance(LineSegment3D_F64 seg,
								   Point3D_F64 pt) {

		// Vector from segment start (a) to point (pt)
		double vecAPtX = pt.x - seg.a.x;
		double vecAPtY = pt.y - seg.a.y;
		double vecAPtZ = pt.z - seg.a.z;

		// Vector representing the line segment (from a to b)
		double vecSegX = seg.b.x - seg.a.x;
		double vecSegY = seg.b.y - seg.a.y;
		double vecSegZ = seg.b.z - seg.a.z;

		// Calculate the squared length of the segment
		double segmentLengthSq = vecSegX * vecSegX + vecSegY * vecSegY + vecSegZ * vecSegZ;

		// If the segment is a point (length is zero), return the distance from pt to seg.a
		if (segmentLengthSq == 0.0) {
			return pt.distance(seg.a);
		}

		// Calculate the dot product of vecAPt and vecSeg
		// This is (APt . AB)
		double dotProduct = vecAPtX * vecSegX + vecAPtY * vecSegY + vecAPtZ * vecSegZ;

		// Calculate the projection parameter t
		// t = (APt . AB) / |AB|^2
		double t = dotProduct / segmentLengthSq;

		double closestX, closestY, closestZ;

		// Clamp t to [0, 1] to find the closest point on the segment
		if (t < 0.0) {
			// Closest point is seg.a
			closestX = seg.a.x;
			closestY = seg.a.y;
			closestZ = seg.a.z;
		} else if (t > 1.0) {
			// Closest point is seg.b
			closestX = seg.b.x;
			closestY = seg.b.y;
			closestZ = seg.b.z;
		} else {
			// Closest point is on the segment, interpolate
			// P_closest = A + t * (B - A)
			closestX = seg.a.x + t * vecSegX;
			closestY = seg.a.y + t * vecSegY;
			closestZ = seg.a.z + t * vecSegZ;
		}

		// Calculate the squared distance from the point pt to the closest point on the segment
		double distSq = (pt.x - closestX) * (pt.x - closestX) +
						(pt.y - closestY) * (pt.y - closestY) +
						(pt.z - closestZ) * (pt.z - closestZ);

		// Return the square root of the squared distance
		return Math.sqrt(distSq);
	}