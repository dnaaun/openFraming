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
