public static double distance(LineSegment3D_F64 seg,
								   Point3D_F64 pt) {

		double dx = pt.x - seg.a.x;
		double dy = pt.y - seg.a.y;
		double dz = pt.z - seg.a.z;

		double cc = dx*dx + dy*dy + dz*dz;

		double vx = seg.b.x - seg.a.x;
		double vy = seg.b.y - seg.a.y;
		double vz = seg.b.z - seg.a.z;

		double len = (double) Math.sqrt(vx*vx + vy*vy + vz*vz);

		double t = (vx*dx + vy*dy + vz*dz) / len;

		// check end points
		if( t <= 0 )
			return pt.distance(seg.a);
		else if( t >= len )
			return pt.distance(seg.b);

		double distSq = cc - t*t;

		// round off error can make distanceSq go negative when it is very close to zero
		if( distSq < 0 ) {
			return 0;
		} else {
			return Math.sqrt(distSq);
		}
	}