
package JavaRecommendationSystem;

/**
 *
 * @author adeshp
 */
import scala.Tuple2;
import org.apache.spark.api.java.*;
import org.apache.spark.mllib.recommendation.ALS;
import org.apache.spark.mllib.recommendation.MatrixFactorizationModel;
import org.apache.spark.mllib.recommendation.Rating;
import org.apache.spark.SparkConf;

public class JavaRecommendationSystem{
    public static void main(String[] args){
        // Need to initialize the spark configuration by giving name
        SparkConf conf = new SparkConf().setAppName("Example Recommendation System")
                .setMaster("local[*]");
        // Create a spark context using the sparkConf
        JavaSparkContext jsc = new JavaSparkContext(conf);

        // Get the data
        String path = "/Users/yapmodeveloper/Downloads/test.txt";
        
        ///create the RDD for this data
        JavaRDD<String> data = jsc.textFile(path);

        // map the data into the desired form
        //Here we are getting the ratings data and convert it into 
        //RDD

        JavaRDD<Rating> ratings = data.map(s -> {
            // split the data of comma seperated values to string array
            String[] sarray = s.split(",");
           
            //get the Rating object to pass to the ALS model

            return new Rating(Integer.parseInt(sarray[0]), Integer.parseInt(sarray[1]),
                                Double.parseDouble(sarray[2]));

        });

        //Now that we have the data let's build the model
        // we will us ethe ALS alogorithm to create the Model.

        int rank = 10;
        int numInterations = 10;
        // create the model by passing above parameters
        MatrixFactorizationModel model = ALS.train(JavaRDD.toRDD(ratings), rank, numInterations, 0.01);
        // model is created

        //Let's test the model and evaluation
        JavaRDD<Tuple2<Object, Object>> userProducts=
        ratings.map(r -> new Tuple2<>(r.user(), r.product()));

        JavaPairRDD<Tuple2<Integer, Integer>, Double> predictions = JavaPairRDD.fromJavaRDD(
            model.predict(JavaRDD.toRDD(userProducts)).toJavaRDD()
            .map(r -> new Tuple2<>(new Tuple2<>(r.user(), r.product()), r.rating()))
        );

        JavaRDD<Tuple2<Double, Double>> ratesAndPreds = JavaPairRDD.fromJavaRDD(
            ratings.map(r -> new Tuple2<>(new Tuple2<>(r.user(), r.product()), r.rating())))
            .join(predictions).values();
        
        //check the model correctness

        double MSE = ratesAndPreds.mapToDouble(pair ->{
            double err = pair._1() - pair._2();
            return err *err;
        }).mean();

        //Print the model accuracy test of MSE

        System.out.println("Mean Squared Error= " + MSE);

        //Save the model
        model.save(jsc.sc(), "/Users/yapmodeveloper/Downloads/myRecommendationSystemUsingCollaborativeApproach");

        MatrixFactorizationModel loadModel = MatrixFactorizationModel.load(jsc.sc(), "/Users/yapmodeveloper/Downloads/myRecommendationSystemUsingCollaborativeApproach");

        // Now get a test data and predict usisng this model.

        jsc.stop();
    }
}
