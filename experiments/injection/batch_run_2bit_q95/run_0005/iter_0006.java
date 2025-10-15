public static double distance(LineSegment3D_F64 l, Point3D_F64 p) {
  // Vector from A to P
  double dx = p.x - l.a.x;
  double dy = p.y - l.a.y;
  double dz = p.z - l.a.z;

  // Squared distance from A to P
  double ap2 = dx * dx + dy * dy + dz * dz;

  // Vector from A to B
  double vx = l.b.x - l.a.x;
  double vy = l.b.y - l.a.y;
  double vz = l.b.z - l.a.z;

  // Length squared of AB
  double l2 = vx * vx + vy * vy + vz * vz;
  double len = Math.sqrt(l2);

  // Degenerate segment check
  if (l2 == 0.0) {
    return Math.sqrt(ap2);
  }

  // Projection of AP onto AB, in units of AB length
  double t = (vx * dx + vy * dy + vz * dz) / len;

  // Check end points
  if (t <= 0.0) {
    return Math.sqrt(ap2);
  } else if (t >= len) {
    double bdx = p.x - l.b.x;
    double bdy = p.y - l.b.y;
    double bdz = p.z - l.b.z;
    return Math.sqrt(bdx * bdx + bdy * bdy + bdz * bdz);
  }

  double distanceSq = ap2 - t * t;

  // round off error can make distanceSq go negative when it is very close to zero
  if (distanceSq < 0.0) {
    return 0.0;
  } else {
    return Math.sqrt(distanceSq);
  }
}