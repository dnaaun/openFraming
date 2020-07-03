NOTE: You will have to prefix every endpoint with `/api/`.  If the server is running on `http://localhost:5000`, the first endpoint below can be accessed by doing `http://localhost:5000/api/classifiers`,


## List all classifiers.
### Endpoint
`GET /classifiers/`

### Request body
Empty. 

### Return body 
#### When successful
```python
[
  { 
     "classifier_id": int, 
     "classifier_name": str,
     "provided_by_openFraming": bool,
     "category_names": [str, ...],
     "status": One_of("not_begun", "training", "completed"),
     "metrics": One_of(
        NULL,
        {
	  "macro_f1_score": float,
	  "macro_precision": float,
	  "macro_recall": float,
	  "accuracy": float
        }
      )
   },
  ...
]
```

### Status
Done. 


## Creates a classifier.
### Endpoint
`POST /classifiers/`

### Request body
```python
{
  "classifier_name": str,
  "category_names": [str, ...]
}
```

### Return body 
#### When successful
```python
{
  "classifier_id": int,
  "policy_issue_name": str,
  "provided_by_openFraming": False,
  "frame_names": [str, ...],
  "status": "not_begun"
}
```

### Status
Done.


## Get details about one classifier.
### Endpoint
`GET /classifiers/<classifier_id:int>`

### Request body
Empty.


### Return body 
#### When successful
```python
{ 
  "classifier_id": int, 
  "classifier_name": str,
  "provided_by_openFraming": bool,
  "category_names": [str, ...],
  "status": One_of("not_begun", "training", "completed")
  "metrics": One_of(
     NULL,
     {
       "macro_f1_score": float,
       "macro_precision": float,
       "macro_recall": float,
       "accuracy": float
     }
   )
}
```

### Status
Done.


## Upload training data for a classifier and start training.
### Endpoint
`POST /classifiers/<classifier_id:int>/training/file`

### Request body
`FormData` 
 `file`: A CSV/Excel file.

### Return body 
#### When successful
```python
{ 
  "classifier_id": int, 
  "classifier_name": str,
  "provided_by_openFraming": False,
  "category_names": [str, ...],
  "status": "training",
  "metrics": NULL
}
```

### Status
Done.

## Lists all test sets for a classifier.
### Endpoint
`GET /classifiers/<classifier_id:int>/test_sets/`

### Request body
Empty.

### Return body 
#### When successful
```python
[
  {
    "classifier_id": int,
    "test_set_id": int,
    "test_set_name": str,
    "inference_status": One_of("not_begun", "predicting", "completed")
  } ...
]
```

### Status
Done.

## Create a test set.
### Endpoint
`POST /classifiers/<classifier_id:int>/test_sets/`

### Request body
```python
{ 
  "test_set_name": str, # Some name the user wants to provide for this set
}
```

### Return body 
#### When successful
```python
{ 
  "classifier_id": int,
  "test_set_id": int,
  "test_set_name": str,
  "status": "not_begun"
}
``` 

### Status
Done.

## Get details about one test set.
### Endpoint
`GET /classifiers/<classifier_id:int>/test_sets/<test_set_id:int>`

### Request body
Empty.


### Return body 
 #### When successful
```python
{
  "classifier_id": int,
  "test_set_id": int
  "test_set_name": str,
  "inference_status": One_of("not_begun", "predicting", "completed")
}
```

### Status
In progress.


## Upload test set and start inference.
### Endpoint
`POST /classifiers/<classifier_id:int>/test_sets/<test_set_id:int>/file`


### Request body
`FormData`
1. `file`

### Return body 
#### When successful
```python
{ 
  "classifier_id": int,
  "test_set_id": int,
  "test_set_name": str,
  "status": "predicting"
}
```

### Status
Done.


## Download predictions on test set.
### Endpoint
`GET /classifiers/<classifier_id:int>/test_sets/<test_set_id:int>/predictions`

### Request parameters
1. `file_type`: Optional. Can be:`csv`, `xlsx`.

### Request body
Empty.

### Return body 
#### When successful
A CSV/Excel file with the predictions.


### Status
In progress.



# List all topic models.
### Endpoint
`GET /topic_models/`

### Request body
Empty.

### Return body 
#### When successful
```python
[
  {
    "topic_model_id": int,
    "topic_model_name": str,
    "num_topics": int,
    "topic_names": One_of(NULL, [str, ...])
    "status": One_of("not_begun", "training", "topics_to_be_named", "completed")
  },
  ...
]
```

### Status
 Done



## Creates a topic model.
### Endpoint
`POST /topic_models/`

### Request body
```python
{
  "topic_model_name": str,
  "num_topics": int
}
```

### Return body 
#### When successful
```python
{
  "topic_model_id": int # The id of the topic detector just created
  "topic_model_name": str,
  "num_topics": int,
  "topic_names": NULL, 
  "status": "not_begun"
 }
```

### Status
Done.


## Get details about one topic model.
### Endpoint
`GET /topic_models/<topic_model_id:int>`

### Request body 
Empty.

### Return body 
#### When successful
```python
{
  "topic_model_id": int,
  "topic_model_name": str,
  "num_topics": int,
  "topic_names": One_of(NULL, [str, ...]),
  "status": One_of("not_begun", "training", "topics_to_be_named", "completed")
}
```


### Status
In progress.

## Upload training file and start training topic model.
### Endpoint
`POST /topic_models/<topic_model_id:int>/training/file`

### Request body
`FormData`

1. `file`

### Return body 
#### When successful
```python
{
  "topic_model_id": int,
  "topic_model_name": str,
  "num_topics": int,
  "topic_names": NULL,
  "status":  "training"
}
```

### Status
Done

### Endpoint
`GET /topic_models/<topic_model_id:int>/topics/preview`

### Request body
Empty.

### Return body 
#### When successful
```python
{
  "topic_model_id": int 
  "topic_model_name": str,
  "num_topics": int,
  "topic_names": One_of(NULL, [str, ...]),
  "status":  One_of("topics_to_be_named", "completed"),
  "topic_previews": [
    {
      "keywords": [ str, ... ],
      "examples": [str, ... ]
    },
    ...  # The length of this list is equal to "num_topics"
  ]
}
```

### Status
Done.


## Name the topics of a trained topic model.
### Endpoint
`POST /topic_models/<topic_model_id:int>/topics/names`

### Request body
```python
{ 
    "topic_names": [str, ...] # Length must be equal to "num_topics" of topic model.
}
```

### Return body 
 #### When successful
```python
{
     "topic_model_id": int, 
     "topic_model_name": str,
     "num_topics": int,
     "topic_names": [str, ...],
     "status": "completed"
}
```

### Status
Done

## Download keywords per each topic.
### Endpoint
`GET /topic_models/<topic_model_id:int>/keywords`

### Request parameters
1. `file_type`: Optional. Can be:`csv`, `xlsx`.

### Request body
Empty.

### Return body 
#### When successful
A `csv` or `xlsx` file.


### Status
In progress.

## Download topic proportions for each example.
### Endpoint
`GET /topic_models/<topic_model_id:int>/keywords`

### Request parameters
1. `file_type`: Optional. Can be:`csv`, `xlsx`.

### Request body
Empty.

### Return body 
#### When successful
A `csv` or `xlsx` file.

### Status
In progress.
