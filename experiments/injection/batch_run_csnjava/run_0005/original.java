public static double distance( LineSegment3D_F64 l,
								   Point3D_F64 p ) {

		double dx = p.x - l.a.x;
		double dy = p.y - l.a.y;
		double dz = p.z - l.a.z;

		double cc = dx*dx + dy*dy + dz*dz;

		double slope_x = l.b.x - l.a.x;
		double slope_y = l.b.y - l.a.y;
		double slope_z = l.b.z - l.a.z;

		double n = (double) Math.sqrt(slope_x*slope_x + slope_y*slope_y + slope_z*slope_z);

		double d = (slope_x*dx + slope_y*dy + slope_z*dz) / n;

		// check end points
		if( d <= 0 )
			return p.distance(l.a);
		else if( d >= n )
			return p.distance(l.b);

		double distanceSq = cc-d*d;

		// round off error can make distanceSq go negative when it is very close to zero
		if( distanceSq < 0 ) {
			return 0;
		} else {
			return Math.sqrt(distanceSq);
		}
	}