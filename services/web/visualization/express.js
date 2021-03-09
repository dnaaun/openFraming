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
    res.sendFile(__dirname + '/Datasets/category-brands.csv');
});

router.get('/covid19_US.csv', function(req, res){
  res.sendFile(__dirname + '/Datasets/covid19_US.csv');
});

router.get('/covid19_KO.csv', function(req, res){
  res.sendFile(__dirname + '/Datasets/covid19_KO.csv');
});

router.get('/gunviolence_data.csv', function(req, res){
  res.sendFile(__dirname + '/Datasets/gunviolence_data.csv');
});
//add the router
app.use('/', router);
app.listen(process.env.port || 3000);

console.log('Running at Port 3000');