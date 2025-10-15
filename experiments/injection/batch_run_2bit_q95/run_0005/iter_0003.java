public static double distance(LineSegment3D_F64 l, Point3D_F64 p) {
    // Vector from A to P
    double dx = p.x - l.a.x;
    double dy = p.y - l.a.y;
    double dz = p.z - l.a.z;

    // Squared distance from A to P
    double ap2 = dx*dx + dy*dy + dz*dz;

    // Vector from A to B
    double vx = l.b.x - l.a.x;
    double vy = l.b.y - l.a.y;
    double vz = l.b.z - l.a.z;

    // Length of AB
    double len = Math.sqrt(vx*vx + vy*vy + vz*vz);

    // Degenerate segment check
    if (len == 0.0) {
        return p.distance(l.a);
    }

    // Projection of AP onto AB, in units of AB length
    double t = (vx*dx + vy*dy + vz*dz) / len;

    // Check end points
    if (t <= 0.0) {
        return p.distance(l.a);
    } else if (t >= len) {
        return p.distance(l.b);
    }

    double distanceSq = ap2 - t*t;

    // round off error can make distanceSq go negative when it is very close to zero
    if (distanceSq < 0.0) {
        return 0.0;
    } else {
        return Math.sqrt(distanceSq);
    }
}