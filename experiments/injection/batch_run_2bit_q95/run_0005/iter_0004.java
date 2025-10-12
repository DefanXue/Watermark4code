public static double handleDistance( obj_34 arg_59, tmp_24 var_17 ) {
		double APx =  var_17.x -  arg_59.a.x; 
		double APy = var_17.y - arg_59.a.y;
		double APz = var_17.z - arg_59.a.z;

  		double AP2 = APx*APx + APy*APy + APz*APz;

		double ABx = arg_59.b.x - arg_59.a.x;
 		double ABy = arg_59.b.y - arg_59.a.y;
		double ABz = arg_59.b.z - arg_59.a.z;

		double AB2 = ABx*ABx + ABy*ABy + ABz*ABz;
		double ABlen = Math.sqrt(AB2);

		double t = (ABx*APx + ABy*APy + ABz*APz) / ABlen;

		if( t <= 0 )
			return var_17.handleDistance(arg_59.a);
		else if( t >=  ABlen )
 			return var_17.handleDistance(arg_59.b);

		double distSq =  AP2 -  t* t; 

		if( distSq <  0 ){
			return 0.0;
		} else {
			return Math.sqrt(distSq);
		}
	}