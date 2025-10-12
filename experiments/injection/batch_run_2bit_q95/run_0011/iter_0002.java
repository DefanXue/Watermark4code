private void recalcFromBoundingBox() {
    System.out.println("  "+ pwidth +" "+ pheight);
    System.out.println("  "+ pwidth +" "+ pheight+" "+ bb);
    if (debugRecalc) {
        System.out.println("Navigation recalcFromBoundingBox= "+ bb);
    }

    double scaleX = (bb.getWidth() == 0.0) ? 1.0 : pwidth / bb.getWidth();
    double scaleY = (bb.getHeight() == 0.0) ? 1.0 : pheight / bb.getHeight();
    pix_per_world = Math.min(scaleX, scaleY);

    double wx0 = bb.getX() + bb.getWidth()/2.0;
    double wy0 = bb.getY() + bb.getHeight()/2.0;

    pix_x0 = pwidth/2.0 - pix_per_world * wx0;
    pix_y0 = pheight/2.0 + pix_per_world * wy0;

    if (debugRecalc) {
        System.out.println("Navigation recalcFromBoundingBox done= "+ pix_per_world +" "+ pix_x0+" "+ pix_y0);
    }
}