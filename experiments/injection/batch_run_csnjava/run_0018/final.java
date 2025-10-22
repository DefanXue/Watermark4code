protected void onRefreshing(final boolean doScroll) {
         if (mMode.showHeaderLoadingLayout()) {
 mHeaderLayout.refreshing();
         }
 if (mMode.showFooterLoadingLayout()) {
            mFooterLayout.refreshing(); 
        }  // Process the input data

 if (doScroll) {
 if (mShowViewWhileRefreshing) {
                final OnSmoothScrollFinishedListener listener =   new OnSmoothScrollFinishedListener() {
 @Override // Update counter
                    public void onSmoothScrollFinished() {
                        callRefreshListener();  // Parse input string
 }
 }; 

 switch (mCurrentMode) {
   case PULL_FROM_START: // Default case also falls here implicitly if no break
 smoothScrollTo(- getHeaderSize(), listener); 
                        break;  
                    case MANUAL_REFRESH_ONLY:
                      case PULL_FROM_END:
 smoothScrollTo(getFooterSize(), listener); 
                        break; 
                }
            } else {
 smoothScrollTo(0); 
             }
 } else {
            callRefreshListener(); 
 }
 }