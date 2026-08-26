import grpc
from concurrent import futures
import os

import doclens_pb2
import doclens_pb2_grpc

from run_pipeline import run_pipeline_on_image

class DocumentAnalysisServicer(doclens_pb2_grpc.DocumentAnalysisServicer):

    def AnalyzeDocument(self, request, context):
        print(f"Received request — document_id: {request.document_id}, filename: {request.filename}")

        script_dir = os.path.dirname(os.path.abspath(__file__))
        temp_path = os.path.join("..", "data", "converted", f"_incoming_{request.document_id or 'temp'}.png")

        try:
            with open(temp_path, "wb") as f:
                f.write(request.image_data)

            result = run_pipeline_on_image(temp_path)

        except Exception as e:
            print(f"Pipeline error: {e}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"Pipeline failed: {str(e)}")
            return doclens_pb2.DocumentReport()

        return doclens_pb2.DocumentReport(
            document_id=request.document_id,
            risk_level=result["risk_level"],
            reason=result["reason"],
            full_report=result["report"]
        )


def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
    doclens_pb2_grpc.add_DocumentAnalysisServicer_to_server(
        DocumentAnalysisServicer(), server
    )
    server.add_insecure_port('[::]:50051')
    server.start()
    print("DocLens AI gRPC server running on port 50051...")
    server.wait_for_termination()


if __name__ == "__main__":
    serve()