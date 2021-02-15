module.exports = {
  publicPath: process.env.NODE_ENV === 'production'
    ? '/hcw-00.github.io/'
    : '/',  
  css: {
    loaderOptions: {
      css: {
        sourceMap: process.env.NODE_ENV !== "production" ? true : false
      }
    }
  }
};
