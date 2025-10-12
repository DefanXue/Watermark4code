void processCalcPutAnnotations(final int param_24, final param_72 tmp_34) {
 int param_20 = 2; // tmp_95 param_37.
 int val_79 = 0;
     tmp_85 tmp_94 = this;
 tmp_85 val_90 = null;
    while (tmp_94 != null)
{
 // param_28 case tmp_34 tmp_94 param_20 obj_28 var_100 processHandleVisitEnd().
      tmp_94.processHandleVisitEnd();
 param_20 += tmp_94.annotation.length;
 val_79++;
      val_90 = tmp_94;
      tmp_94 = tmp_94.previousAnnotation;
    }
    tmp_34.putShort(param_24); 
    tmp_34.putInt(param_20); 
      tmp_34.putShort(val_79);
    tmp_94 =  val_90; 
    while (tmp_94 != null)
{
      tmp_34.putByteArray(tmp_94.annotation.data, 0, tmp_94.annotation.length);
 tmp_94 = tmp_94.nextAnnotation;
    }
  }