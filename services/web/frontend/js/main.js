$(".other-option").hide();

$(".form-check-inline").on("click", function(){
   $(".other-option").fadeOut();
});

$(".other-policy").on("click", function(){
   $(".other-option").fadeIn()();
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

	//
	// HANDLEBARS STARTS HERE
	// 
	var BASE_URL = "http://ec2-3-90-135-165.compute-1.amazonaws.com/api";
	var GET_ONE_TOPIC_MDL_URL = BASE_URL + "/topic_models/<TOPIC_MODEL_ID>";

	var one_topic_mdl_url = make_url(GET_ONE_TOPIC_MDL_URL, [ ["<TOPIC_MODEL_ID>", "2"] ]) 
	$.ajax({
		url: one_topic_mdl_url,
		type: 'GET',
		dataType: 'json',
		success: render_one_topic_mdl,
		error: handle_ajax_error
	})
	

	
	//
	// HANDLEBARS ENDS HERE
	//



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
