private void recalcFromBoundingBox() {
      System.out.println("  "+ pwidth +" "+ pheight);
      System.out.println("  "+ pwidth +" "+ pheight+" "+ bb);
    if (debugRecalc) {
 System.out.println("Navigation recalcFromBoundingBox= "+ bb);
      }

      / /  decide which dimension is limiting
     double pixx_per_wx = (bb.getWidth() == 0.0) ? 1 : pwidth / bb.getWidth();
 double pixy_per_wy = (bb.getHeight() == 0.0) ? 1 : pheight / bb.getHeight();
 pix_per_world = Math.min(pixx_per_wx, pixy_per_wy);
  
      // calc the center point
    double wx0 = bb.getX() + bb.getWidth()/2;
     double wy0 = bb.getY() + bb.getHeight()/2;

 // calc offset based on center point
    pix_x0 =  pwidth/ 2 -  pix_per_world *  wx0; 
 pix_y0 = pheight/2 + pix_per_world * wy0;

 if (debugRecalc) {
 System.out.println("Navigation recalcFromBoundingBox done= "+ pix_per_world +" "+ pix_x0+" "+ pix_y0);
    }
 }