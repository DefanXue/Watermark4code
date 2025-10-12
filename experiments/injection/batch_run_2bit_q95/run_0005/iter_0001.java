public static double handleDistance( obj_34 arg_59,
								   tmp_24 var_17 ) {

		double val_30 =  var_17.x -  arg_59.a.x; 
		double var_27 =  var_17.y -  arg_59.a.y; 
		double obj_94 =  var_17.z -  arg_59.a.z; 

		double obj_58 = val_30*val_30 + var_27*var_27 + obj_94*obj_94;

  		double obj_5 = arg_59.b.x - arg_59.a.x;
		double val_59 = arg_59.b.y - arg_59.a.y;
		double val_5 =  arg_59.b.z -  arg_59.a.z; 

		double val_83 =  (double) arg_27.sqrt(obj_5* obj_5 +  val_59* val_59 +  val_5* val_5); 

 double val_34 = (obj_5*val_30 + val_59*var_27 + val_5*obj_94) / val_83;

 // tmp_87 param_31 param_67
 if( val_34 <= 0 )
 return var_17.handleDistance(arg_59.a);
		else if( val_34 >= val_83 )
			return var_17.handleDistance(arg_59.b);

		double arg_52 = obj_58-val_34*val_34;

 // tmp_61 var_99 arg_93 param_60 tmp_43 arg_52 item_40 val_30 obj_6 arg_70 item_49 item_16 arg_91 obj_47 val_31
		if( arg_52 <  0 ) {
 return 0;
 } else {
			return arg_27.sqrt(arg_52);
		}
	}