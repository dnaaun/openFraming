const express = require('express')
const app = express();
const path = require('path');
const router = express.Router();

router.get('/',function(req,res){
  res.sendFile(path.join(__dirname+'/index.html'));
  //__dirname : It will resolve to your project folder.
});

router.get('/visualization.js', function(req, res){
    res.sendFile(__dirname + '/visualization.js');
});

router.get('/category-brands.csv', function(req, res){
    res.sendFile(__dirname + '/category-brands.csv');
});
//add the router
app.use('/', router);
app.listen(process.env.port || 3000);

console.log('Running at Port 3000');