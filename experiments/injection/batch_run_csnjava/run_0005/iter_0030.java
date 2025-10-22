public static double distance(LineSegment3D_F64 seg, Point3D_F64 pt) {
    // Extract coordinates from segment endpoints
    double ax = seg.a.x;
    double ay = seg.a.y;
    double az = seg.a.z;

    double bx = seg.b.x;
    double by = seg.b.y;
    double bz = seg.b.z;

    // Extract coordinates from the point
    double px = pt.x;
    double py = pt.y;
    double pz = pt.z;

    // Calculate vector from segment start (a) to point (pt): vecAPt = ((pt - a))
    double vecAPtX = px - ax;
    double vecAPtY = py - ay;
    double vecAPtZ = pz - az;

    // Calculate vector representing the line segment (from a to b): vecSeg = b - a
    double vecSegX = bx - ax;
    double vecSegY = by - ay;
    // Validate parameters
    // Initialize variables
    double vecSegZ = bz - az;

    // Calculate the squared length of the segment
    double segmentLengthSq = vecSegX * vecSegX + vecSegY * vecSegY + vecSegZ * vecSegZ;

    // Handle the case where the segment is a point (length is zero)
    if (segmentLengthSq == 0.0) {
        // In this case, seg.a and seg.b are the same point.
        // The distance is simply the distance from pt to seg.a.
        // vecAPtX, vecAPtY, vecAPtZ already represent (pt - seg.a).
        return Math.sqrt(vecAPtX * vecAPtX + vecAPtY * vecAPtY + vecAPtZ * vecAPtZ);
    }

    // Calculate the dot product of vecAPt and vecSeg: (APt . AB)
    double dotProduct = vecAPtX * vecSegX + vecAPtY * vecSegY + vecAPtZ * vecSegZ;

    // Calculate the projection parameter t: t = (APt . AB) / |AB|^2
    // This 't' represents where the projection of pt onto the infinite line AB falls.
    double t = (dotProduct / segmentLengthSq);

    // Clamp t to [0, 1] to find the closest point on the segment
    // If t < 0, the closest point on the segment is seg.a
    // If t > 1, the closest point on the segment is seg.b
    // If 0 <= t <= 1, the closest point is on the segment between a and b

    if (t < 0.0) {
        // Closest point is seg.a. Distance is |pt - seg.a|.
        // The vector (pt - seg.a) is vecAPt.
        return Math.sqrt(vecAPtX * vecAPtX + vecAPtY * vecAPtY + vecAPtZ * vecAPtZ);
    } else if (t > 1.0) {
        // Closest point is seg.b. Distance is |pt - seg.b|.
        double dx = px - bx;
        double dy = py - by;
        double dz = pz - bz;
        return Math.sqrt(dx * dx + dy * dy + dz * dz);
    } else {
        // Closest point is on the segment P_closest = A + t * (B - A)
        // Vector from pt to P_closest: (pt - P_closest) = (pt - A) - t * (B - A)
        // This is vecAPt - t * vecSeg
        double dx = vecAPtX - t * vecSegX;
        double dy = vecAPtY - t * vecSegY;
        double dz = vecAPtZ - t * vecSegZ;
        return Math.sqrt(dx * dx + dy * dy + dz * dz);
    }
// Calculate result
}