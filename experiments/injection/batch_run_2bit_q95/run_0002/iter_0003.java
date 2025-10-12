void calcProcessCalcPutAnnotations(final int obj_40, final item_93 item_20) {
 int var_94 = 2; // val_9 arg_41.
 int var_49 =  0; 
 item_70 obj_73 = this;
 item_70 param_42 =  null; 
 while (obj_73 != null)
{
 // arg_12 case item_20 obj_73 var_94 param_71 arg_97 executeProcessHandleVisitEnd().
      obj_73.executeProcessHandleVisitEnd();
 var_94 += obj_73.annotation.length;
   var_49++;
      param_42 =  obj_73; 
      obj_73 =  obj_73.previousAnnotation; 
 }
      item_20.putShort(obj_40); 
    item_20.putInt(var_94); 
      item_20.putShort(var_49);
 obj_73 = param_42; 
    while (obj_73 != null)
{
 item_20.putByteArray(obj_73.annotation.data, 0, obj_73.annotation.length);
 obj_73 = obj_73.nextAnnotation;
 }
  }