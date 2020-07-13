//jshint esversion:8
$('#result').hide();
$(".other-option").hide();

$(".form-check-inline").on("click", function(){
   $(".other-option").fadeOut();
   $('#other_text').val('')
   $('#category_names').val('')
});

$(".other-policy").on("click", function(){
   $(".other-option").fadeIn()();
});




var stateClassifier_id='';
var stateStatus='';
var email='';
var testName='';
var testId='';
var resultText='';


// GET POST
//

// const endpoint = "http://ec2-3-90-135-165.compute-1.amazonaws.com/api/"
const endpoint = "http://www.openframing.org/api/"
// const endpoint = "http://localhost/api/"


async function getFraming() {
	// console.log("asdfdas");
	var endpointGET = endpoint + 'classifiers/';
	await axios
	   .get(endpointGET)
	   .then(res => console.log(res))
	   .catch(err => console.error(err))
 }
   
async function postFraming() {
	console.log("postFraming");
	// let name = $("input[name='policyissue']:checked").val();



	
	var name = $("input[name='policyissue']:checked").val();
	if (name =="other") {
		name = document.getElementById("other_text").value;
	}

	// if (name === "gunviolence") {
	// 	console.log("gunviolence")
	// 	var arrayCategoryNames = ["gun_topic1", "gun_topic2"]
	// 	console.log(arrayCategoryNames)
	// } else if (name === "immigration") {
	// 	console.log("immigration")
	// 	var arrayCategoryNames = ["immigration1", "immigration2"]
	// 	console.log(arrayCategoryNames)
	// } else if (name === "tobocco") {
	// 	console.log("tobocco")
	// 	var arrayCategoryNames = ["tobocco1", "tobocco2"]
	// 	console.log(arrayCategoryNames)
	// } else if (name === "samesexmarriage") {
	// 	console.log("samesexmarriage")
	// 	var arrayCategoryNames = ["samesexmarriage1", "samesexmarriage2"]
	// 	console.log(arrayCategoryNames)
	// } else if (name === "covid19") {
	// 	console.log("covid19")
	// 	var arrayCategoryNames = ["covid19_1", "covid19_2"]
	// 	console.log(arrayCategoryNames)
	// } else if (name === "climatechange") {
	// 	console.log("climatechange")
	// 	var arrayCategoryNames = ["climatechange1", "climatechange2"]
	// 	console.log(arrayCategoryNames)
	// } else if (name === "other") {
	// 	console.log("other")
	// 	var stringCategoryNames = document.getElementById("category_names").value;
	// 	var arrayCategoryNames = stringCategoryNames.split(',');
	// 	console.log(arrayCategoryNames)
	// } 


	// console.log(name);
	
	var endpointPOST = endpoint + 'classifiers/';
	await axios
		
		.post(endpointPOST, {
			name: name,
			category_names: arrayCategoryNames,
			notify_at_email: email
		})
		.then(res => {
			var classifier_id = res.data["classifier_id"];
			console.log(classifier_id)
			stateClassifier_id = classifier_id;
			// return classifier_id
		})
		.catch(err => console.error(err))

	// var classifier_id = res.data["classifier_id"]
	// console.log(classifier_id);
	// callback();
	
}

//
//GET /classifiers/<classifier_id:int>
//
function checkClassifier(classifier_id) {
	var endpointGET = endpoint + 'classifiers/' + classifier_id
	axios
	   .get(endpointGET)
	   .then(res => console.log(res.data))
	   .catch(err => console.error(err));
	
}


async function upTrainingFile() {

	var endpointUploadTraining = endpoint + 'classifiers/' + stateClassifier_id + '/training/file'

	var formData = new FormData();
	var imagefile = document.querySelector('#annotatedsamplefile1');
	formData.append("file", imagefile.files[0]);
	console.log(imagefile.files[0])

	await axios
		.post(endpointUploadTraining, formData, {
			headers: {
				'Content-Type': 'multipart/form-data'
			}
		})
	    .then(res => {
			console.log(res);
			
		})
		.catch(err => {
	        console.error({err});
	    });
}



async function checkClassifier() {
	// var a = 3

	var endpointCheckClassifier = endpoint + 'classifiers/' + stateClassifier_id

	await axios
		.get(endpointCheckClassifier)
		.then(res => {
			var stateStatus = res.data["status"]
			// console.log(stateStatus)
			if (stateStatus == "training") {
				console.log(stateStatus);
				// postTestName();
			} else if (stateStatus == "completed") {
				clearInterval(i);
				afterTraining();
			}


		})

		.catch(err => {
			console.error({err});
		});
}

var i = null;
async function looping() {
	i = setInterval(function(){
		checkClassifier();
	}, 5000);
}



async function getFraming() {
	console.log("getFraming");
	var endpointGET = endpoint + 'classifiers/'
	await axios
	   .get(endpointGET)
	   .then(res => console.log(res))
	   .catch(err => console.error(err));
	
 }



async function postTestName() {
	console.log("postTestName");
	var endpointPostTestName = endpoint + 'classifiers/' + stateClassifier_id + '/test_sets/'
	// var endpointPostTestName = endpoint + 'classifiers/' + 40 + '/test_sets/'
	console.log(endpointPostTestName);
	console.log(testName);
	await axios
		.post(endpointPostTestName, {
			test_set_name: testName,
			notify_at_email: email
		})
		.then(res => console.log(res))
		.catch(err => console.error(err))
	// getTestId();
}

async function getTestId() {
	console.log("getTestId");
	var endpointGetTestId = endpoint + 'classifiers/' + stateClassifier_id + '/test_sets/'
	await axios
		

		.get(endpointGetTestId)
		.then(res => {
			// console.log(res);
			// console.log(res.data);
			// console.log(res.data.length);
			var i;
			for (i = 0; i < res.data.length; i++) {
				console.log(res.data[i]["test_set_name"]);
				if (testName == res.data[i]["test_set_name"]) {
					testId = res.data[i]["test_set_id"]
					console.log(testId);
				}
			}


		})
		.catch(err => console.error(err));
	// upTestingFile();
}

async function upTestingFile() {
	console.log("upTestingFile");

	var endpointUploadTraining = endpoint + 'classifiers/' + stateClassifier_id + '/test_sets/' + testId + '/file'


	var formData = new FormData();
	var imagefile = document.querySelector('#annotatedsamplefile2');
	formData.append("file", imagefile.files[0]);
	console.log(imagefile.files[0])

	await axios
		.post(endpointUploadTraining, formData, {
			headers: {
				'Content-Type': 'multipart/form-data'
			}
		})
	    .then(res => {
	        console.log({res});
		})
		.catch(err => {
	        console.error({err});
		});
	// getPred();
}

var i = null;
var stopper = false
async function loopingTesting() {
	
	i = setInterval(function(){
		// checkTesting();
		console.log(stopper);
		if (stopper == false) {
			checkTesting();
			
		} else if (stopper == true) {
			clearInterval(i);
		}
		
	}, 5000);
}

async function checkTesting() {
	// var a = 3

	var endpointCheckTesting = endpoint + 'classifiers/' + stateClassifier_id + '/test_sets/' + testId;

	await axios
		.get(endpointCheckTesting)
		.then(res => {
			// var stateStatus = res.data["status"]
			// console.log(stateStatus)
			// console.log(res);
			// console.log(res.data);
			// console.log(res.data[testId-1]);
			// console.log(res.data[testId-1]["status"]);
			statusTesting = res.data["status"]
			if (statusTesting == "completed") {
				stopper = true
				console.log(stopper);
				getPred()
				
				
			} else {
				console.log("Please wait");
			}

		})

		.catch(err => {
			console.error({err})
		})
}

async function getPred() {
	console.log("getPred");
	var endpointGetPred = endpoint + 'classifiers/' + stateClassifier_id + '/test_sets/' + testId + '/predictions?file_type=csv';
	await axios
		

		.get(endpointGetPred)
		.then(res => {
			console.log(res)
			$('#progressBar').addClass("w-100");
			$('#progressBar').removeClass("progress-bar-striped progress-bar-animated")
			resultText = res.data;
			showresult();
			$('#download-button').removeClass("disabled")
		})
		.catch(err => console.error(err))
}





var fileName = $('input[type="file"]').change(function(e){
	var fileName = e.target.files[0].name;
	console.log('The file "' + fileName +  '" has been selected.');
}); // collect file name after uploaded


async function initTraining() {
	await postFraming(); // return classifier id
	await getFraming();
	$('#progressBar').addClass("w-25");
	await upTrainingFile();
	await looping();
}

async function afterTraining() {
	$('#progressBar').addClass("w-75");
	await postTestName(); // return test set id
	await getTestId();
	await upTestingFile();
	await loopingTesting();

	// await checkTesting();
	// await getPred();
}

async function noTraining() {
	$('#progressBar').addClass("w-75");
	await postTestName(); // return test set id
	await getTestId();
	await upTestingFile();
	await loopingTesting();

	// await checkTesting();
	// await getPred();
}


async function showresult() {
	$('#result-status').text("Completed! Here's the result.");
	document.getElementById('result_text').value = resultText;
}

var stateImage2=false;
var stateEmail=false;

function validateEmail(email) {
	var emailReg = /^([\w-\.]+@([\w-]+\.)+[\w-]{2,4})?$/;
	return emailReg.test(email);
  }

var emailAddress = $("#email");

// if(!validateEmail(emailAddress)) { 
// 	stateEmail=true
// } else {
// 	stateEmail=false
// 	// $('#email').on("change", function(){
		
// 	// })
// }

$('#annotatedsamplefile2').on("change", function(){ 
	$('#performAnalysis').click(async function(){

		// if(!validateEmail(emailAddress)) { 
			/* do stuff here */ 
			console.log(name);
			console.log(emailAddress)
			$('#hideStep1').hide();
			$('#hideStep1_1').hide();
			$('#hideStep2').hide();
			$('#hideStep3').hide();
			
			$('#result').fadeIn();
			
			email = document.getElementById("email").value;
			testName = document.getElementById("other_text").value;
			policyIssue = $('input[name=policyissue]:checked', '#policyissueradiobutton').val();
		
			if (policyIssue == "gunviolence") {
				stateClassifier_id = 1;
				testName = policyIssue;
				noTraining();
			} else if (policyIssue == "other") {
				await initTraining();
			}
		// }


	 });
});


 




/* submit button

$(function () {
    $('form').on('submit', function (e) {

        $.ajax(
                        {
                            url:'/submitData',
                            data: $('form').serialize(),
                            type: 'POST',
                            success: function (data) {
                                alert("Submitted");
                                console.log(data);
                        },
                        error: function (error){
                            console.log(error)
                        }
                    });
    });

});

 */

// The API endpoint prefix
var API_PREFIX = "/api";

function make_url(template_url, list_of_subst) {
	// tempelate_url: str
	// list_of_subst: [ (to_remove: str, to_add: str), ....]
	var built_url = template_url;

	for(let i=0; i < list_of_subst.length; i++) {
		var to_remove = list_of_subst[i][0];
		var to_add = list_of_subst[i][1];
		var built_url = built_url.replace(to_remove, to_add);
	}
	return built_url
}

function handle_ajax_error() {
}

function render_one_topic_mdl(data, textStatus, jqxhr) {
	// data: JSON Object
	// textStatus: string
	// jqxhr:  jqXHR Object
	//
	var one_topic_mdl_source = document.getElementById("topic-model-template").innerHTML;
	var one_topic_mdl_template = Handlebars.compile(one_topic_mdl_source);
	var one_topic_mdl_html = one_topic_mdl_template(data); 
	var all_topic_mdls_container = $('#all_topic_models');
	all_topic_mdls_container.append(one_topic_mdl_html);
}



$(function(){


	$("#otherradiobutton").on('click', function () {

		let hasError = false;
		let color = 'red';

		if ($("annotatedsamplefile").val() == null){
			alert('null');
			$('#emptysamplefileerror').show();
			$('#emptysamplefileerror').css('color', color);
			hasError = true;
		}
		else {
			$('#emptysamplefileerror').hide();
			hasError = false;
		}


		$('button[type="submit"]').prop('disabled', hasError);

	})


	function validate() {
		$(".error, #commaerror").hide();
		let hasError = false;

		// validate category names
		$('#category_names').each(function() {
			if ($(this).val().indexOf(",") == -1) {
				//alert('Please separate multiple keywords with a comma.');
				$('#commaerror').show();
				hasError = true;
			}
		});
		// validate email
		let emailReg = /^([\w-\.]+@([\w-]+\.)+[\w-]{2,4})?$/;
		let emailAddressVal = $('#email').val();
		if (emailAddressVal == '') {
			$("#email").after('<span class="error">Please enter your email address.</span>');
			hasError = true;
		}
		else if (!emailReg.test(emailAddressVal)) {
			$("#email").after('<span class="error">Enter a valid email address.</span>');
			hasError = true;
		}

		return hasError;
	}

	$("#category_names, #email").on('keyup', function(event) {
		$('button[type="submit"]').prop('disabled', validate());
	});



	$("button[type='submit']").on('click', function (){
		let radioValue = $("input[name='policyissue']:checked").val();
		alert(radioValue);
		console.log(radioValue);
		debugger;
	})

	$("#create_a_classifier_form").on('submit',  function(e){
		// alert("Submitted");
		let form = $(this);

		let name = form.find('#name').val();
		let raw_category_names = form.find('#category_names').val();

		let url = API_PREFIX + '/classifiers';

		let category_names = raw_category_names.split(',');
		let json_data = {
		 	"name": name,
		 	"category_names": category_names
		}
		let data = JSON.stringify(json_data);
		//debugger;

		$.ajax(
			{
				url: url,
				data: data,
				type: 'POST',
			        contentType: 'application/json; charset=utf-8',
			        dataType: 'json',
				success: function (data, textStatus, jqXHR) {
					alert("Submitted successfully.");
					console.log(data);
				},
				error: function (jqXHR, textStatus, errorThrown){
					console.log("THERE WAS AN AJAX ERROR");
					console.log(jqXHR.status);
					console.log(jqXHR);
					console.log(textStatus);
					console.log(errorThrown);
				}
			})

		// alert(data);
    });
});


function otherIssue(){
	let a = document.getElementById('other_fruit');
	a.checked=true;
}
function regularFruit(){
	let a = document.getElementById('other_text');
	a.value="";
}

