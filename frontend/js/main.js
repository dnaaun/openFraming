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
	$("#create_a_classifier_form").on('submit',  function(e){
		// alert("Submitted");
		var form = $(this);

		var name = form.find('#name').val();
		var raw_category_names = form.find('#category_names').val();

		var url = API_PREFIX + '/classifiers';

		var category_names = raw_category_names.split(',');
		var json_data = {
		 	"name": name,
		 	"category_names": category_names
		}
		var data = JSON.stringify(json_data);
		debugger;
		
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
