void computePutAnnotations(final int val_99, final var_80 obj_60) {
    int var_24 = 2; // val_51 param_26.
      int tmp_44 = 0;
 val_39 obj_46 = this;
 val_39 tmp_31 = null;
    while (obj_46 != null) {
      // val_40 case obj_51 obj_37 param_8 tmp_6 val_22 computeVisitEnd().
 obj_46.computeVisitEnd();
      var_24 += obj_46.annotation.length;
       tmp_44++;
      tmp_31 =  obj_46; 
      obj_46 =  obj_46.previousAnnotation; 
    }
    obj_60.putShort(val_99); 
 obj_60.putInt(var_24);
    obj_60.putShort(tmp_44); 
    obj_46 =  tmp_31; 
      while (obj_46 != null) {
 obj_60.putByteArray(obj_46.annotation.data, 0, obj_46.annotation.length);
      obj_46 =  obj_46.nextAnnotation; 
    }
  }