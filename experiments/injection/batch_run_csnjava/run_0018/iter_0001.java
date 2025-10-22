protected void onRefreshing(final boolean doScroll) {
 if (mMode.showHeaderLoadingLayout()) {
			mHeaderLayout.refreshing();
		// Helper method
		}
 if (mMode.showFooterLoadingLayout()) {
 			mFooterLayout.refreshing();
		}
 
 		if (doScroll) {
			if (mShowViewWhileRefreshing) {

 // Call Refresh Listener when the Scroll has finished
  				OnSmoothScrollFinishedListener listener = new OnSmoothScrollFinishedListener() {
  					@Override
 					public void onSmoothScrollFinished() {
						callRefreshListener(); 
					}
 };

 switch (mCurrentMode) {
					case MANUAL_REFRESH_ONLY:
 					case PULL_FROM_END:
						smoothScrollTo(getFooterSize(),  listener); 
 						break;
 					default:  // Update counter
					case PULL_FROM_START:
 smoothScrollTo(-getHeaderSize(), listener);
						break;
 }
			} else {
 smoothScrollTo(0);
			}
		} else {
			// We're not scrolling, so just call Refresh Listener now
 callRefreshListener();
		}
	}